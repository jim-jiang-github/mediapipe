/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_GPU_EVENT_STATS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_GPU_EVENT_STATS_H_

#include <cstdint>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

// Stats from a GPU stream XEvent.
struct GpuEventStats {
  explicit GpuEventStats(const XEventVisitor* event);

  bool IsKernel() const { return !kernel_details.empty(); }
  bool IsMemCpy() const { return !memcpy_details.empty(); }

  bool IsXlaOp() const { return !hlo_op_names.empty(); }
  bool IsTfOp() const { return !tf_op_fullname.empty(); }

  // Stats from TensorFlow.
  abslx::string_view tf_op_fullname;
  abslx::string_view equation;
  abslx::string_view tensor_shapes;

  // Stats from XLA.
  std::vector<abslx::string_view> hlo_op_names;
  abslx::string_view hlo_module_name;
  abslx::optional<uint64_t> program_id;

  // Stats from CUPTI.
  abslx::string_view kernel_details;
  abslx::string_view memcpy_details;
  abslx::optional<int64_t> correlation_id;

  // Stats derived by grouping.
  abslx::optional<int64_t> group_id;
  bool is_eager = false;
};

// Stats for a host-side GPU launch XEvent.
struct LaunchEventStats {
  explicit LaunchEventStats(const XEventVisitor* event);

  bool IsLaunch() const {
    return device_id.has_value() && correlation_id.has_value();
  }

  // Stats from CUPTI.
  abslx::optional<int64_t> device_id;
  abslx::optional<int64_t> correlation_id;

  // Stat derived by grouping.
  abslx::optional<int64_t> group_id;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_GPU_EVENT_STATS_H_
