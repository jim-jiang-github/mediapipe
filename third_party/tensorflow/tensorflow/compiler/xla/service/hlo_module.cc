/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/compilation_environments.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/mapped_ptr_container_sorter.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace xla {

HloModule::HloModule(const std::string& name, HloModuleConfig config)
    : HloModule(name, config, std::make_unique<CompilationEnvironments>()) {}

Status HloModule::set_schedule(HloSchedule schedule) {
  TF_RET_CHECK(schedule.module() == this);
  TF_RETURN_IF_ERROR(schedule.Verify());
  schedule_ = std::move(schedule);
  return OkStatus();
}

void HloModule::ReplaceEntryComputation(HloComputation* entry_computation) {
  entry_computation_ = entry_computation;
  config_.SetDefaultComputationLayout(
      entry_computation_->ComputeProgramShape());
  input_output_alias_config_ = HloInputOutputAliasConfig(
      entry_computation_->root_instruction()->shape());
}

HloComputation* HloModule::AddComputationInternal(
    std::unique_ptr<HloComputation> computation, bool is_entry,
    bool uniquify_identifiers, bool preserve_entry_layouts) {
  if (is_entry) {
    CHECK_EQ(nullptr, entry_computation_);
    entry_computation_ = computation.get();

    if (preserve_entry_layouts) {
      config_.SetComputationLayoutIfExists(
          entry_computation_->ComputeProgramShape());
    } else if (!config_.has_entry_computation_layout()) {
      // If the module configuration has no entry layout computation set, create
      // a default one based on the program shape.
      config_.SetDefaultComputationLayout(
          entry_computation_->ComputeProgramShape());
    }
    input_output_alias_config_ = HloInputOutputAliasConfig(
        entry_computation_->root_instruction()->shape());
  }

  if (uniquify_identifiers) {
    computation->UniquifyName(&computation_name_uniquer_);
    for (auto* instruction : computation->instructions()) {
      instruction->UniquifyName(&instruction_name_uniquer_);
    }

    // Pick unique IDs for each instruction.
    for (auto* instruction : computation->instructions()) {
      instruction->SetUniqueId(NewUniqueInstructionId());
    }
    // Set unique id to this computation.
    CHECK_NE(computation->root_instruction()->unique_id(), -1)
        << "Root has no valid id: " << computation->ToString();
    computation->SetUniqueId(computation->root_instruction()->unique_id());
  } else {
    // Don't uniquify the names of the computation or instruction, but we must
    // run the names through the uniquifiers to prevent future name collisions
    // for computations and instructions created later. Also, set the
    // next_unique_id_ to the one greater than the max unique id of any
    // instruction (or the computation) to avoid ID collisions.
    computation_name_uniquer_.GetUniqueName(computation->name());
    for (auto* instruction : computation->instructions()) {
      instruction_name_uniquer_.GetUniqueName(instruction->name());
      next_unique_id_ = std::max(next_unique_id_, instruction->unique_id() + 1);
    }
    if (next_unique_id_ < computation->unique_id() + 1) {
      next_unique_id_ = computation->unique_id() + 1;
    }
  }

  computation->set_parent(this);
  computations_.push_back(std::move(computation));
  return computations_.back().get();
}

HloComputation* HloModule::AddEntryComputation(
    std::unique_ptr<HloComputation> computation) {
  return AddComputationInternal(std::move(computation), /*is_entry=*/true,
                                /*uniquify_identifiers=*/true,
                                /*preserve_entry_layouts=*/false);
}

HloComputation* HloModule::AddEntryComputationWithLayouts(
    std::unique_ptr<HloComputation> computation) {
  return AddComputationInternal(std::move(computation), /*is_entry=*/true,
                                /*uniquify_identifiers=*/true,
                                /*preserve_entry_layouts=*/true);
}

Status HloModule::RemoveEmbeddedComputation(HloComputation* to_remove) {
  if (has_schedule() && !to_remove->IsCalledComputation()) {
    schedule_->remove_computation(to_remove);
  }

  auto it = abslx::c_find_if(
      computations_, [&to_remove](const std::unique_ptr<HloComputation>& comp) {
        return comp.get() == to_remove;
      });
  TF_RET_CHECK(it != computations_.end());
  TF_RET_CHECK(it->get() == to_remove);
  computations_.erase(it);
  return OkStatus();
}

HloComputation* HloModule::AddEmbeddedComputation(
    std::unique_ptr<HloComputation> computation) {
  return AddComputationInternal(std::move(computation), /*is_entry=*/false,
                                /*uniquify_identifiers=*/true,
                                /*preserve_entry_layouts=*/false);
}

void HloModule::ReplaceComputations(
    const abslx::flat_hash_map<HloComputation*, HloComputation*>& replacements) {
  // Replace all uses of non-canonical computations with their
  // representatives.
  std::vector<std::unique_ptr<HloComputation>> new_computations;
  new_computations.reserve(computations_.size());

  for (std::unique_ptr<HloComputation>& computation : computations_) {
    for (auto* instruction : computation->instructions()) {
      switch (instruction->opcode()) {
        case HloOpcode::kAllReduce:
        case HloOpcode::kCall:
        case HloOpcode::kMap:
        case HloOpcode::kReduce:
        case HloOpcode::kReduceScatter:
        case HloOpcode::kReduceWindow:
        case HloOpcode::kScatter:
        case HloOpcode::kSort: {
          HloComputation* new_arg = tensorflow::gtl::FindWithDefault(
              replacements, instruction->to_apply(), nullptr);
          if (new_arg != nullptr) {
            instruction->set_to_apply(new_arg);
          }
          break;
        }
        case HloOpcode::kWhile: {
          HloComputation* new_condition = tensorflow::gtl::FindWithDefault(
              replacements, instruction->while_condition(), nullptr);
          if (new_condition != nullptr) {
            instruction->set_while_condition(new_condition);
          }
          HloComputation* new_body = tensorflow::gtl::FindWithDefault(
              replacements, instruction->while_body(), nullptr);
          if (new_body != nullptr) {
            instruction->set_while_body(new_body);
          }
          break;
        }
        case HloOpcode::kConditional: {
          for (int b = 0; b < instruction->branch_count(); ++b) {
            HloComputation* new_computation = tensorflow::gtl::FindWithDefault(
                replacements, instruction->branch_computation(b), nullptr);
            if (new_computation != nullptr) {
              instruction->set_branch_computation(b, new_computation);
            }
          }
          break;
        }
        case HloOpcode::kSelectAndScatter: {
          HloComputation* new_select = tensorflow::gtl::FindWithDefault(
              replacements, instruction->select(), nullptr);
          if (new_select != nullptr) {
            instruction->set_select(new_select);
          }
          HloComputation* new_scatter = tensorflow::gtl::FindWithDefault(
              replacements, instruction->scatter(), nullptr);
          if (new_scatter != nullptr) {
            instruction->set_scatter(new_scatter);
          }
          break;
        }
        default:
          break;
      }
    }

    if (replacements.find(computation.get()) == replacements.end()) {
      new_computations.push_back(std::move(computation));
    }
  }

  // Replace entry_computation if necessary.
  entry_computation_ = tensorflow::gtl::FindWithDefault(
      replacements, entry_computation_, entry_computation_);

  computations_ = std::move(new_computations);
}

std::string HloModule::ToString(const HloPrintOptions& options) const {
  return std::string(ToCord(options));
}

abslx::Cord HloModule::ToCord(const HloPrintOptions& options) const {
  abslx::Cord result;
  result.Append("HloModule ");
  if (options.print_ids()) {
    // When print_ids() is false, exclude module's name because it includes and
    // leads to non-deterministic fingerprint.
    result.Append(name());
  }
  if (has_schedule()) {
    TF_CHECK_OK(schedule().Verify());
    result.Append(", is_scheduled=true");
  }
  std::string serialized_aliasing = input_output_alias_config().ToShortString();
  if (!serialized_aliasing.empty()) {
    result.Append(", input_output_alias={ ");
    result.Append(std::move(serialized_aliasing));
    result.Append(" }");
  }
  if (config_.alias_passthrough_params()) {
    result.Append(", alias_passthrough_params=true");
  }
  if (config_.has_entry_computation_layout()) {
    result.Append(", entry_computation_layout={");
    result.Append(entry_computation_layout().ToString());
    result.Append("}");
  }
  if (config_.allow_spmd_sharding_propagation_to_output()) {
    result.Append(", allow_spmd_sharding_propagation_to_output=true");
  }
  result.Append("\n\n");
  const auto& computations = options.canonicalize_computations()
                                 ? MakeComputationSorted()
                                 : MakeComputationPostOrder();
  for (const HloComputation* computation : computations) {
    // Don't print async computations when the sytax sugar is enabled since that
    // is redundant information.
    if (options.syntax_sugar_async_ops() && computation->IsAsyncComputation()) {
      continue;
    }
    if (computation == entry_computation()) {
      result.Append("ENTRY ");
    }
    if (has_schedule() && schedule().is_computation_scheduled(computation)) {
      result.Append(computation->ToCord(
          options, schedule().sequence(computation).instructions()));
    } else {
      result.Append(computation->ToCord(options));
    }
    result.Append("\n\n");
  }
  return result;
}

HloModuleProto HloModule::ToProto() const {
  HloModuleProto proto;
  proto.set_id(unique_id_);
  proto.set_name(name_);
  if (entry_computation_) {
    proto.set_entry_computation_name(entry_computation_->name());
    proto.set_entry_computation_id(entry_computation_->unique_id());
    *proto.mutable_host_program_shape() =
        entry_computation_layout().ComputeProgramShape().ToProto();
  }
  for (const HloComputation* computation : MakeComputationPostOrder()) {
    HloComputationProto computation_proto = computation->ToProto();
    proto.add_computations()->Swap(&computation_proto);
  }
  if (has_schedule()) {
    *proto.mutable_schedule() = schedule().ToProto().ValueOrDie();
  }
  *proto.mutable_input_output_alias() = input_output_alias_config().ToProto();
  *proto.mutable_dynamic_parameter_binding() =
      dynamic_parameter_binding().ToProto();
  for (const auto& parameter_indices : CrossProgramPrefetches()) {
    const auto& parameter = parameter_indices.first;
    const auto& indices = parameter_indices.second;
    auto* prefetch = proto.mutable_cross_program_prefetches()->Add();
    prefetch->set_parameter(parameter);
    for (auto index : indices) {
      prefetch->add_index(index);
    }
  }
  proto.set_is_dynamic(is_dynamic_);
  if (has_spmd_output_sharding()) {
    *proto.mutable_spmd_output_sharding() = spmd_output_sharding().ToProto();
  }

  if (has_spmd_parameters_shardings()) {
    for (const auto& parameter_sharding : spmd_parameters_shardings()) {
      *proto.add_spmd_parameters_shardings() = parameter_sharding.ToProto();
    }
  }

  proto.set_use_auto_spmd_partitioning(use_auto_spmd_partitioning_);

  for (const HloModuleProto::ProfileInfo& profile_info : profile_info_list_) {
    HloModuleProto::ProfileInfo& profile_info_proto =
        *proto.mutable_profile_info()->Add();
    profile_info_proto.set_profile_type(profile_info.profile_type());
    profile_info_proto.set_relative_speedup(profile_info.relative_speedup());
    profile_info_proto.set_profile_source(profile_info.profile_source());
    profile_info_proto.set_compilation_event(profile_info.compilation_event());
  }
  if (this->config_.has_static_device_assignment()) {
    DeviceAssignmentProto device_assignment;
    TF_CHECK_OK(
        this->config_.static_device_assignment().Serialize(&device_assignment));
    (*proto.mutable_device_assignment()) = device_assignment;
  }
  return proto;
}

Status HloModule::CheckUniqueNamesAndIdsForComputationsAndInstructions() const {
  abslx::flat_hash_set<std::string> computation_names;
  abslx::flat_hash_set<int> computation_ids;
  abslx::flat_hash_set<std::string> instruction_names;
  abslx::flat_hash_set<int> instruction_ids;

  for (const HloComputation* computation : computations()) {
    TF_RET_CHECK(!ContainsKey(computation_names, computation->name()))
        << "Computation name is not unique: " << computation->name();
    computation_names.insert(computation->name());

    TF_RET_CHECK(!ContainsKey(computation_ids, computation->unique_id()))
        << "Computation id is not unique: " << computation->unique_id();
    computation_ids.insert(computation->unique_id());

    for (const HloInstruction* instruction : computation->instructions()) {
      TF_RET_CHECK(!ContainsKey(instruction_names, instruction->name()))
          << "Instruction name is not unique: " << instruction->name();
      instruction_names.insert(instruction->name());

      TF_RET_CHECK(!ContainsKey(instruction_ids, instruction->unique_id()))
          << "Instruction id is not unique: " << instruction->unique_id();
      instruction_ids.insert(instruction->unique_id());
    }
  }
  return OkStatus();
}

/* static */
StatusOr<std::unique_ptr<HloModule>> HloModule::CreateFromProto(
    const HloModuleProto& proto, const HloModuleConfig& module_config,
    bool prohibit_empty_literal) {
  VLOG(2) << "CreateFromProto()";
  XLA_VLOG_LINES(3, proto.DebugString());

  // The ProgramShape in the passed in module config must match the shapes of
  // the entry parameters and root.
  TF_RET_CHECK(proto.has_host_program_shape())
      << "No program shape found in the proto";
  ProgramShape expected_program_shape(proto.host_program_shape());
  TF_RET_CHECK(expected_program_shape.parameters_size() ==
               module_config.entry_computation_layout().parameter_count());
  for (int i = 0; i < expected_program_shape.parameters_size(); ++i) {
    const Shape& parameter_shape =
        module_config.entry_computation_layout().parameter_layout(i).shape();
    TF_RET_CHECK(ShapeUtil::Compatible(expected_program_shape.parameters(i),
                                       parameter_shape))
        << "HloModuleConfig has different shape for parameter " << i
        << " than the HLO module. Expected: "
        << ShapeUtil::HumanStringWithLayout(
               expected_program_shape.parameters(i))
        << ", actual: " << ShapeUtil::HumanStringWithLayout(parameter_shape);
  }
  const Shape& result_shape =
      module_config.entry_computation_layout().result_layout().shape();
  TF_RET_CHECK(
      ShapeUtil::Compatible(expected_program_shape.result(), result_shape))
      << "HloModuleConfig has different result shape than the HLO module. "
         "Expected: "
      << ShapeUtil::HumanStringWithLayout(expected_program_shape.result())
      << ", actual: " << ShapeUtil::HumanStringWithLayout(result_shape);

  abslx::flat_hash_map<int64_t, HloComputation*> computation_map;
  abslx::flat_hash_map<HloComputation*, int64_t> to_proto_id;
  std::vector<std::unique_ptr<HloComputation>> computations;
  HloComputation* entry = nullptr;
  for (const HloComputationProto& computation_proto : proto.computations()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloComputation> computation,
        HloComputation::CreateFromProto(computation_proto, computation_map,
                                        prohibit_empty_literal));
    CHECK_NE(computation.get(), nullptr);
    int64_t computation_id = computation_proto.id();
    TF_RET_CHECK(computation_id != -1);
    TF_RET_CHECK(!ContainsKey(computation_map, computation_id));
    computation_map[computation_id] = computation.get();
    to_proto_id[computation.get()] = computation_id;
    if (computation_id == proto.entry_computation_id()) {
      entry = computation.get();
    }
    computations.push_back(std::move(computation));
  }
  TF_RET_CHECK(entry != nullptr);

  auto module = std::make_unique<HloModule>(proto.name(), module_config);

  // Sort the computations in the proto id's order.
  abslx::c_sort(computations, [&](const std::unique_ptr<HloComputation>& a,
                                 const std::unique_ptr<HloComputation>& b) {
    return to_proto_id[a.get()] < to_proto_id[b.get()];
  });

  // Add sorted computations to the module.
  for (auto& computation : computations) {
    bool is_entry = computation.get() == entry;
    // Don't uniquify names because we want names to be stable across
    // serialization and deserialization.
    module->AddComputationInternal(std::move(computation), is_entry,
                                   /*uniquify_identifiers=*/false,
                                   /*preserve_entry_layouts=*/false);
  }
  TF_RET_CHECK(module->entry_computation_ != nullptr);
  TF_ASSIGN_OR_RETURN(
      module->input_output_alias_config_,
      HloInputOutputAliasConfig::CreateFromProto(
          entry->ComputeProgramShape().result(), proto.input_output_alias()));

  // Because we didn't uniquify the names or the ids, double-check that the
  // instruction and computation names and ids are unique from the proto.
  TF_ASSIGN_OR_RETURN(module->dynamic_parameter_binding_,
                      DynamicParameterBinding::CreateFromProto(
                          proto.dynamic_parameter_binding()));

  TF_RETURN_IF_ERROR(
      module->CheckUniqueNamesAndIdsForComputationsAndInstructions());

  if (proto.has_schedule()) {
    TF_ASSIGN_OR_RETURN(
        HloSchedule schedule,
        HloSchedule::CreateFromProto(module.get(), proto.schedule()));
    TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));
  }

  for (const auto& prefetch : proto.cross_program_prefetches()) {
    module->AddCrossProgramPrefetch(
        prefetch.parameter(),
        ShapeIndex(prefetch.index().begin(), prefetch.index().end()));
  }

  module->set_is_dynamic(proto.is_dynamic());

  if (proto.has_spmd_output_sharding()) {
    TF_ASSIGN_OR_RETURN(HloSharding hlo_sharding,
                        HloSharding::FromProto(proto.spmd_output_sharding()));
    module->set_spmd_output_sharding(hlo_sharding);
  }

  std::vector<HloSharding> param_shardings;
  for (const auto& sharding_proto : proto.spmd_parameters_shardings()) {
    TF_ASSIGN_OR_RETURN(HloSharding sharding,
                        HloSharding::FromProto(sharding_proto));
    param_shardings.push_back(sharding);
  }
  if (!param_shardings.empty()) {
    module->set_spmd_parameters_shardings(param_shardings);
  }

  module->set_use_auto_spmd_partitioning(proto.use_auto_spmd_partitioning());

  for (const auto& profile_info : proto.profile_info()) {
    module->add_profile_info(profile_info);
  }
  if (proto.has_device_assignment()) {
    if (!module->config_.has_static_device_assignment()) {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<DeviceAssignment> device_assignment,
          DeviceAssignment::Deserialize(proto.device_assignment()));
      module->config_.set_static_device_assignment(*device_assignment);
    }
  }
  return std::move(module);
}

/* static */
StatusOr<HloModuleConfig> HloModule::CreateModuleConfigFromShape(
    const ProgramShape& program_shape, const DebugOptions& debug_options,
    const ExecutionOptions* execution_options) {
  HloModuleConfig module_config(ProgramShape{program_shape});
  module_config.set_debug_options(debug_options);
  if (execution_options) {
    if (execution_options->num_replicas() > 0) {
      module_config.set_replica_count(execution_options->num_replicas());
    }
    if (execution_options->num_partitions() > 0) {
      module_config.set_num_partitions(execution_options->num_partitions());
    }
    module_config.set_use_spmd_partitioning(
        execution_options->use_spmd_partitioning());
    module_config.set_use_auto_spmd_partitioning(
        execution_options->use_auto_spmd_partitioning());
    std::vector<int64_t> mesh_shape;
    for (auto t : execution_options->auto_spmd_partitioning_mesh_shape()) {
      mesh_shape.push_back(t);
    }
    module_config.set_auto_spmd_partitioning_mesh_shape(mesh_shape);
    std::vector<int64_t> mesh_ids;
    for (auto t : execution_options->auto_spmd_partitioning_mesh_ids()) {
      mesh_ids.push_back(t);
    }
    module_config.set_auto_spmd_partitioning_mesh_ids(mesh_ids);
    module_config.set_deduplicate_hlo(execution_options->deduplicate_hlo());
    module_config.set_allow_spmd_sharding_propagation_to_output(
        execution_options->allow_spmd_sharding_propagation_to_output());
    if (execution_options->has_device_assignment()) {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<DeviceAssignment> device_assignment,
                          DeviceAssignment::Deserialize(
                              execution_options->device_assignment()));
      module_config.set_static_device_assignment(*device_assignment);
      if (execution_options->num_replicas() > 0) {
        CHECK_EQ(module_config.static_device_assignment().replica_count(),
                 module_config.replica_count());
      }
      if (execution_options->num_partitions() > 0) {
        CHECK_EQ(module_config.static_device_assignment().computation_count(),
                 module_config.num_partitions());
      }
    }
  }

  // The module config is constructed with default layouts regardless of what is
  // passed in via the ProgramShape. Set the layouts to the appropriate values.
  ComputationLayout* entry_layout =
      module_config.mutable_entry_computation_layout();
  for (int64_t i = 0; i < entry_layout->parameter_count(); ++i) {
    TF_RETURN_IF_ERROR(
        entry_layout->mutable_parameter_layout(i)->CopyLayoutFromShape(
            program_shape.parameters(i)));
  }
  TF_RETURN_IF_ERROR(entry_layout->mutable_result_layout()->CopyLayoutFromShape(
      program_shape.result()));
  return module_config;
}

/* static */
StatusOr<HloModuleConfig> HloModule::CreateModuleConfigFromProto(
    const HloModuleProto& module, const DebugOptions& debug_options,
    const ExecutionOptions* execution_options) {
  if (!module.has_host_program_shape()) {
    return tensorflow::errors::FailedPrecondition(
        "No program shape found in the proto");
  }
  ProgramShape program_shape(module.host_program_shape());
  TF_ASSIGN_OR_RETURN(HloModuleConfig config,
                      CreateModuleConfigFromShape(program_shape, debug_options,
                                                  execution_options));
  if (!config.has_static_device_assignment()) {
    if (module.has_device_assignment()) {
      // Get the proto from the exeuction options rather than the module proto.
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<DeviceAssignment> device_assignment,
          DeviceAssignment::Deserialize(module.device_assignment()));
      config.set_static_device_assignment(*device_assignment);
    }
  }
  return config;
}

namespace {
// Returns whether `hlo` is used outside the given subcomputation.
// `instructions_in_subcomputation` is the instruction set of the given
// subcomputation.
bool IsUsedOutsideSubcomputation(const HloInstruction& hlo,
                                 const abslx::flat_hash_set<HloInstruction*>&
                                     instructions_in_subcomputation) {
  return abslx::c_any_of(hlo.users(), [&](HloInstruction* user) {
    return !instructions_in_subcomputation.contains(user);
  });
}
}  // anonymous namespace

HloInstruction* HloModule::OutlineExpressionFromComputation(
    abslx::Span<HloInstruction* const> instructions_to_outline,
    const std::string& outlined_computation_name, HloComputation* computation) {
  auto builder = HloComputation::Builder(outlined_computation_name);

  // A map from original instructions to their counterparts in the new outlined
  // function.
  abslx::flat_hash_map<HloInstruction*, HloInstruction*> outlined_instructions;
  // A set that contains all instructions to be outlined.
  abslx::flat_hash_set<HloInstruction*> instruction_set_to_outline(
      instructions_to_outline.begin(), instructions_to_outline.end());
  std::vector<HloInstruction*> arguments;
  std::vector<HloInstruction*> outputs;
  int64_t parameter_count = 0;
  for (HloInstruction* instruction_to_outline : instructions_to_outline) {
    // Clone the original instruction.
    HloInstruction* outlined_instruction =
        builder.AddInstruction(instruction_to_outline->Clone());

    // Replace its operands to their counterparts in the new function.
    for (int64_t operand_num = 0;
         operand_num < outlined_instruction->operand_count(); ++operand_num) {
      HloInstruction* old_operand =
          outlined_instruction->mutable_operand(operand_num);

      HloInstruction** operand_slot = &(outlined_instructions[old_operand]);
      if (*operand_slot == nullptr) {
        // Because instructions_to_outline is in topological order, if
        // old_operand is not in outlined_instructions, old_operand must be an
        // input of the outlined subcomputation and thus should be represented
        // as a parameter in the new function.
        arguments.push_back(old_operand);
        *operand_slot = builder.AddInstruction(HloInstruction::CreateParameter(
            parameter_count, old_operand->shape(), "p"));
        ++parameter_count;
      }
      TF_CHECK_OK(
          outlined_instruction->ReplaceOperandWith(operand_num, *operand_slot));
    }

    // Insert the new instruction into the outlined_instructions map.
    InsertOrDie(&outlined_instructions, instruction_to_outline,
                outlined_instruction);

    // Mark instruction_to_outline an output if it is used outside the
    // subcomputation or is the output of the original computation (i.e. used
    // externally).
    if (instruction_to_outline->user_count() == 0 ||
        IsUsedOutsideSubcomputation(*instruction_to_outline,
                                    instruction_set_to_outline)) {
      outputs.push_back(instruction_to_outline);
    }
  }

  if (outputs.size() != 1) {
    std::string error_message =
        "The subcomputation to outline has multiple outputs:\n";
    for (HloInstruction* output : outputs) {
      abslx::StrAppend(&error_message, output->ToString(), "\n");
    }
    LOG(FATAL) << error_message;
  }
  HloInstruction* output = outputs[0];

  // Creates a call to the nested computation.
  HloComputation* nested_computation = AddEmbeddedComputation(
      builder.Build(FindOrDie(outlined_instructions, output)));
  HloInstruction* call = computation->AddInstruction(HloInstruction::CreateCall(
      output->shape(), arguments, nested_computation));

  VLOG(2) << "Outlining the following instructions";
  for (auto* instruction_to_outline : instructions_to_outline) {
    VLOG(2) << "  " << instruction_to_outline->ToString();
  }
  VLOG(2) << "as a call " << call->ToString();
  VLOG(2) << "to " << nested_computation->ToString();

  TF_CHECK_OK(output->ReplaceAllUsesWith(call));
  for (auto i = instructions_to_outline.rbegin();
       i != instructions_to_outline.rend(); ++i) {
    TF_CHECK_OK(computation->RemoveInstruction(*i));
  }

  return call;
}

int64_t HloModule::instruction_count() const {
  int64_t n = 0;
  for (const auto& computation : computations_) {
    n += computation->instruction_count();
  }
  return n;
}

std::vector<HloComputation*> HloModule::MakeComputationPostOrder(
    const abslx::flat_hash_set<abslx::string_view>& execution_threads,
    const abslx::flat_hash_set<HloComputation*>& allow_list) const {
  std::vector<HloComputation*> filtered_post_order(allow_list.size());
  auto post_order = this->MakeComputationPostOrder(execution_threads);

  int filtered_idx = 0;
  for (auto& computation : post_order) {
    if (allow_list.contains(computation)) {
      filtered_post_order[filtered_idx] = computation;
      filtered_idx += 1;
    }
  }

  return filtered_post_order;
}

std::vector<HloComputation*> HloModule::MakeComputationPostOrder(
    const abslx::flat_hash_set<abslx::string_view>& execution_threads) const {
  if (computations_.empty()) {
    return {};
  }
  // First determine all root computations by building a set of nonroot
  // computations (computations which are called by an instruction in the
  // module).
  abslx::flat_hash_set<HloComputation*> nonroot_computations;
  nonroot_computations.reserve(computations_.size() - 1);
  for (auto& computation : computations_) {
    for (auto* instruction : computation->instructions()) {
      for (HloComputation* called_computation :
           instruction->called_computations()) {
        nonroot_computations.insert(called_computation);
      }
    }
  }

  // Keep track of computations which have already been added to the post
  // order. This prevents duplication as an embedded computation may be called
  // from two different root computations.
  abslx::flat_hash_set<HloComputation*> added_computations;
  std::vector<HloComputation*> post_order;
  added_computations.reserve(computations_.size());
  post_order.reserve(computations_.size());
  for (auto& computation : computations_) {
    if (nonroot_computations.contains(computation.get())) {
      continue;
    }
    for (HloComputation* embedded_computation :
         computation->MakeEmbeddedComputationsList()) {
      if (!added_computations.contains(embedded_computation)) {
        post_order.push_back(embedded_computation);
        added_computations.insert(embedded_computation);
      }
    }
    // Root computations should only be encountered once.
    CHECK(!added_computations.contains(computation.get()));
    post_order.push_back(computation.get());
    added_computations.insert(computation.get());
  }
  if (post_order.size() != computations_.size()) {
    for (HloComputation* computation : post_order) {
      LOG(ERROR) << "Post Order: " << computation->name() << " ("
                 << computation->parent()->name() << ")";
    }
    for (auto& computation : computations_) {
      LOG(ERROR) << "Computations: " << computation->name() << " ("
                 << computation->parent()->name() << ")";
    }
    LOG(FATAL) << "Mismatch computation count: post_order=" << post_order.size()
               << " computation_count=" << computations_.size();
  }
  if (execution_threads.empty()) {
    return post_order;
  }
  std::vector<HloComputation*> post_order_with_execution_threads;
  abslx::c_copy_if(
      post_order, std::back_inserter(post_order_with_execution_threads),
      [&execution_threads](HloComputation* computation) {
        return execution_threads.find(computation->execution_thread()) !=
               execution_threads.end();
      });
  return post_order_with_execution_threads;
}

namespace {

class FingerprintMap {
 public:
  void Reserve(int capacity) { fingerprint_map_.reserve(capacity); }

  uint64_t GetFingerprint(const HloComputation* computation) {
    auto result = fingerprint_map_.try_emplace(computation, 0);
    if (result.second) {
      result.first->second =
          tensorflow::Fingerprint64(computation->ToString(print_options_));
    }
    return result.first->second;
  }

 private:
  HloPrintOptions print_options_ = HloPrintOptions::ModuleFingerprint();
  abslx::flat_hash_map<const HloComputation*, uint64_t> fingerprint_map_;
};

void SortComputationsByContent(std::vector<HloComputation*>* computations) {
  FingerprintMap fingerprint_map;
  fingerprint_map.Reserve(computations->size());
  auto cmp = [&fingerprint_map](const HloComputation* a,
                                const HloComputation* b) {
    if (a->instruction_count() != b->instruction_count()) {
      return a->instruction_count() < b->instruction_count();
    }
    return fingerprint_map.GetFingerprint(a) <
           fingerprint_map.GetFingerprint(b);
  };
  abslx::c_sort(*computations, cmp);
}

}  // anonymous namespace

std::vector<HloComputation*> HloModule::MakeComputationSorted(
    const abslx::flat_hash_set<abslx::string_view>& execution_threads) const {
  std::vector<HloComputation*> result =
      MakeComputationPostOrder(execution_threads);
  if (config().content_aware_computation_sorting()) {
    SortComputationsByContent(&result);
  }
  return result;
}

std::vector<HloComputation*> HloModule::MakeNonfusionComputations(
    const abslx::flat_hash_set<abslx::string_view>& execution_threads) const {
  std::vector<HloComputation*> result =
      MakeComputationPostOrder(execution_threads);
  result.erase(std::remove_if(
                   result.begin(), result.end(),
                   [](HloComputation* c) { return c->IsFusionComputation(); }),
               result.end());
  return result;
}

std::vector<HloComputation*> HloModule::MakeNonfusionComputationsSorted(
    const abslx::flat_hash_set<abslx::string_view>& execution_threads) const {
  auto result = MakeNonfusionComputations(execution_threads);
  if (config().content_aware_computation_sorting()) {
    SortComputationsByContent(&result);
  }
  return result;
}

std::unique_ptr<HloModule> HloModule::Clone(const std::string& suffix) const {
  return Clone(config(), suffix);
}

std::unique_ptr<HloModule> HloModule::Clone(const HloModuleConfig& config,
                                            const std::string& suffix) const {
  VLOG(1) << "Cloning module :" << name_ << " --> " << suffix << "\n";
  auto module = abslx::WrapUnique(new HloModule(
      abslx::StrCat(name_, suffix.empty() ? "" : "-", suffix), config,
      std::make_unique<CompilationEnvironments>(*comp_envs_)));

  HloCloneContext context(module.get(), suffix);
  auto cloned_computation = entry_computation_->Clone(suffix, &context);
  module->AddEntryComputation(std::move(cloned_computation));
  module->input_output_alias_config() = input_output_alias_config();
  module->set_is_dynamic(is_dynamic());
  if (has_schedule() && schedule().Verify().ok()) {
    HloSchedule clone_schedule(module.get());
    for (HloComputation* computation : computations()) {
      if (schedule().is_computation_scheduled(computation)) {
        HloComputation* new_computation = context.FindComputation(computation);
        // The module being cloned may have computations that are dead, i.e.,
        // unreachable from the entry computation. In that case, new_computation
        // is nullptr.
        if (new_computation != nullptr) {
          HloInstructionSequence& clone_sequence =
              clone_schedule.GetOrCreateSequence(new_computation);
          for (const HloInstruction* instruction :
               schedule().sequence(computation).instructions()) {
            clone_sequence.push_back(context.GetInstruction(instruction));
          }
        }
      }
    }
    TF_CHECK_OK(module->set_schedule(std::move(clone_schedule)));
  }
  for (const auto& parameter_indices : CrossProgramPrefetches()) {
    const auto& parameter = parameter_indices.first;
    const auto& indices = parameter_indices.second;
    module->AddCrossProgramPrefetch(parameter, indices);
  }

  // To make clone behavior match uncloned behavior, we reorder
  // module->computations_ to match the order in computations_.
  using ComputationSorter = MappedPtrContainerSorter<HloComputation>;
  ComputationSorter::MapPtrFn computation_map_fn =
      [&context](const HloComputation* c) {
        return context.FindComputation(c);
      };
  auto status = ComputationSorter::Sort(
      computation_map_fn, ComputationSorter::IndexAfterMappedElementsFn(),
      computations_, module->computations_);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to sort module computations for " << name() << "; "
               << status;
  }

  return module;
}

Status HloModule::RemoveUnusedComputations() {
  std::string suffix = "tmp";
  auto module = std::make_unique<HloModule>(
      abslx::StrCat(name_, suffix.empty() ? "" : "-", suffix), config());
  HloCloneContext context(module.get(), suffix);
  entry_computation_->Clone(suffix, &context);
  std::vector<HloComputation*> to_remove;
  for (auto computation : computations()) {
    auto found_computation = context.FindComputation(computation);
    if (found_computation == nullptr) {
      to_remove.push_back(computation);
    }
  }
  for (auto computation : to_remove) {
    TF_RETURN_IF_ERROR(RemoveEmbeddedComputation(computation));
  }
  return OkStatus();
}

HloComputation* HloModule::DeepCloneComputation(HloComputation* computation,
                                                HloCloneContext* context) {
  HloComputation* new_computation;
  if (context != nullptr) {
    if ((new_computation = context->FindComputation(computation)) != nullptr) {
      return new_computation;
    }
    new_computation =
        AddEmbeddedComputation(computation->Clone(context->suffix(), context));
  } else {
    new_computation = AddEmbeddedComputation(computation->Clone(""));
  }
  return new_computation;
}

uint64_t HloModule::RandomNew64() const {
  abslx::MutexLock l(&rng_mutex_);
  return rng_();
}

HloComputation* HloModule::GetComputationWithName(abslx::string_view name) {
  auto computations_in_module = computations();
  auto it = abslx::c_find_if(
      computations_in_module,
      [&](HloComputation* computation) { return computation->name() == name; });
  return it == computations_in_module.end() ? nullptr : *it;
}

HloModule::HloModule(const std::string& name, HloModuleConfig config,
                     std::unique_ptr<CompilationEnvironments> comp_envs)
    : name_(NameUniquer::GetSanitizedName(name)),
      config_(std::move(config)),
      unique_id_(next_unique_module_id_++),
      metadata_(tensorflow::Env::Default()),
      comp_envs_(std::move(comp_envs)) {
  metadata_.set_canonical_module_id(unique_id_);
}

/* static */ std::atomic<int> HloModule::next_unique_module_id_(0);

}  // namespace xla
