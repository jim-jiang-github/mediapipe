// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/tensor/tensors_to_segmentation_calculator.proto

#include "mediapipe/calculators/tensor/tensors_to_segmentation_calculator.pb.h"

#include <algorithm>

#include <x/google/protobuf/io/coded_stream.h>
#include <x/google/protobuf/extension_set.h>
#include <x/google/protobuf/wire_format_lite.h>
#include <x/google/protobuf/descriptor.h>
#include <x/google/protobuf/generated_message_reflection.h>
#include <x/google/protobuf/reflection_ops.h>
#include <x/google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <x/google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG
namespace mediapipe {
constexpr TensorsToSegmentationCalculatorOptions::TensorsToSegmentationCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : gpu_origin_(0)

  , activation_(0)

  , output_layer_index_(1){}
struct TensorsToSegmentationCalculatorOptionsDefaultTypeInternal {
  constexpr TensorsToSegmentationCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~TensorsToSegmentationCalculatorOptionsDefaultTypeInternal() {}
  union {
    TensorsToSegmentationCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT TensorsToSegmentationCalculatorOptionsDefaultTypeInternal _TensorsToSegmentationCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto[1];
static const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* file_level_enum_descriptors_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::TensorsToSegmentationCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TensorsToSegmentationCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::TensorsToSegmentationCalculatorOptions, gpu_origin_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TensorsToSegmentationCalculatorOptions, activation_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TensorsToSegmentationCalculatorOptions, output_layer_index_),
  0,
  1,
  2,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 8, sizeof(::mediapipe::TensorsToSegmentationCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_TensorsToSegmentationCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\nEmediapipe/calculators/tensor/tensors_t"
  "o_segmentation_calculator.proto\022\tmediapi"
  "pe\032$mediapipe/framework/calculator.proto"
  "\032\036mediapipe/gpu/gpu_origin.proto\"\342\002\n&Ten"
  "sorsToSegmentationCalculatorOptions\022-\n\ng"
  "pu_origin\030\001 \001(\0162\031.mediapipe.GpuOrigin.Mo"
  "de\022V\n\nactivation\030\002 \001(\0162<.mediapipe.Tenso"
  "rsToSegmentationCalculatorOptions.Activa"
  "tion:\004NONE\022\035\n\022output_layer_index\030\003 \001(\005:\001"
  "1\"0\n\nActivation\022\010\n\004NONE\020\000\022\013\n\007SIGMOID\020\001\022\013"
  "\n\007SOFTMAX\020\0022`\n\003ext\022\034.mediapipe.Calculato"
  "rOptions\030\302\221\276\262\001 \001(\01321.mediapipe.TensorsTo"
  "SegmentationCalculatorOptions"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto_deps[2] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
  &::descriptor_table_mediapipe_2fgpu_2fgpu_5forigin_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto = {
  false, false, 509, descriptor_table_protodef_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto, "mediapipe/calculators/tensor/tensors_to_segmentation_calculator.proto", 
  &descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto_deps, 2, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto(&descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto);
namespace mediapipe {
const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* TensorsToSegmentationCalculatorOptions_Activation_descriptor() {
  ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto);
  return file_level_enum_descriptors_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto[0];
}
bool TensorsToSegmentationCalculatorOptions_Activation_IsValid(int value) {
  switch (value) {
    case 0:
    case 1:
    case 2:
      return true;
    default:
      return false;
  }
}

#if (__cplusplus < 201703) && (!defined(_MSC_VER) || _MSC_VER >= 1900)
constexpr TensorsToSegmentationCalculatorOptions_Activation TensorsToSegmentationCalculatorOptions::NONE;
constexpr TensorsToSegmentationCalculatorOptions_Activation TensorsToSegmentationCalculatorOptions::SIGMOID;
constexpr TensorsToSegmentationCalculatorOptions_Activation TensorsToSegmentationCalculatorOptions::SOFTMAX;
constexpr TensorsToSegmentationCalculatorOptions_Activation TensorsToSegmentationCalculatorOptions::Activation_MIN;
constexpr TensorsToSegmentationCalculatorOptions_Activation TensorsToSegmentationCalculatorOptions::Activation_MAX;
constexpr int TensorsToSegmentationCalculatorOptions::Activation_ARRAYSIZE;
#endif  // (__cplusplus < 201703) && (!defined(_MSC_VER) || _MSC_VER >= 1900)

// ===================================================================

class TensorsToSegmentationCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<TensorsToSegmentationCalculatorOptions>()._has_bits_);
  static void set_has_gpu_origin(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_activation(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static void set_has_output_layer_index(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
};

TensorsToSegmentationCalculatorOptions::TensorsToSegmentationCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.TensorsToSegmentationCalculatorOptions)
}
TensorsToSegmentationCalculatorOptions::TensorsToSegmentationCalculatorOptions(const TensorsToSegmentationCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&gpu_origin_, &from.gpu_origin_,
    static_cast<size_t>(reinterpret_cast<char*>(&output_layer_index_) -
    reinterpret_cast<char*>(&gpu_origin_)) + sizeof(output_layer_index_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.TensorsToSegmentationCalculatorOptions)
}

void TensorsToSegmentationCalculatorOptions::SharedCtor() {
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&gpu_origin_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&activation_) -
    reinterpret_cast<char*>(&gpu_origin_)) + sizeof(activation_));
output_layer_index_ = 1;
}

TensorsToSegmentationCalculatorOptions::~TensorsToSegmentationCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.TensorsToSegmentationCalculatorOptions)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void TensorsToSegmentationCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void TensorsToSegmentationCalculatorOptions::ArenaDtor(void* object) {
  TensorsToSegmentationCalculatorOptions* _this = reinterpret_cast< TensorsToSegmentationCalculatorOptions* >(object);
  (void)_this;
}
void TensorsToSegmentationCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void TensorsToSegmentationCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void TensorsToSegmentationCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.TensorsToSegmentationCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    ::memset(&gpu_origin_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&activation_) -
        reinterpret_cast<char*>(&gpu_origin_)) + sizeof(activation_));
    output_layer_index_ = 1;
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* TensorsToSegmentationCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional .mediapipe.GpuOrigin.Mode gpu_origin = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          ::PROTOBUF_NAMESPACE_ID::uint64 val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
          if (PROTOBUF_PREDICT_TRUE(::mediapipe::GpuOrigin_Mode_IsValid(val))) {
            _internal_set_gpu_origin(static_cast<::mediapipe::GpuOrigin_Mode>(val));
          } else {
            ::PROTOBUF_NAMESPACE_ID::internal::WriteVarint(1, val, mutable_unknown_fields());
          }
        } else goto handle_unusual;
        continue;
      // optional .mediapipe.TensorsToSegmentationCalculatorOptions.Activation activation = 2 [default = NONE];
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          ::PROTOBUF_NAMESPACE_ID::uint64 val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
          if (PROTOBUF_PREDICT_TRUE(::mediapipe::TensorsToSegmentationCalculatorOptions_Activation_IsValid(val))) {
            _internal_set_activation(static_cast<::mediapipe::TensorsToSegmentationCalculatorOptions_Activation>(val));
          } else {
            ::PROTOBUF_NAMESPACE_ID::internal::WriteVarint(2, val, mutable_unknown_fields());
          }
        } else goto handle_unusual;
        continue;
      // optional int32 output_layer_index = 3 [default = 1];
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          _Internal::set_has_output_layer_index(&has_bits);
          output_layer_index_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag == 0) || ((tag & 7) == 4)) {
          CHK_(ptr);
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  _has_bits_.Or(has_bits);
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* TensorsToSegmentationCalculatorOptions::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.TensorsToSegmentationCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional .mediapipe.GpuOrigin.Mode gpu_origin = 1;
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteEnumToArray(
      1, this->_internal_gpu_origin(), target);
  }

  // optional .mediapipe.TensorsToSegmentationCalculatorOptions.Activation activation = 2 [default = NONE];
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteEnumToArray(
      2, this->_internal_activation(), target);
  }

  // optional int32 output_layer_index = 3 [default = 1];
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(3, this->_internal_output_layer_index(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.TensorsToSegmentationCalculatorOptions)
  return target;
}

size_t TensorsToSegmentationCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.TensorsToSegmentationCalculatorOptions)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    // optional .mediapipe.GpuOrigin.Mode gpu_origin = 1;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::EnumSize(this->_internal_gpu_origin());
    }

    // optional .mediapipe.TensorsToSegmentationCalculatorOptions.Activation activation = 2 [default = NONE];
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::EnumSize(this->_internal_activation());
    }

    // optional int32 output_layer_index = 3 [default = 1];
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_output_layer_index());
    }

  }
  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void TensorsToSegmentationCalculatorOptions::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.TensorsToSegmentationCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  const TensorsToSegmentationCalculatorOptions* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<TensorsToSegmentationCalculatorOptions>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.TensorsToSegmentationCalculatorOptions)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.TensorsToSegmentationCalculatorOptions)
    MergeFrom(*source);
  }
}

void TensorsToSegmentationCalculatorOptions::MergeFrom(const TensorsToSegmentationCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.TensorsToSegmentationCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    if (cached_has_bits & 0x00000001u) {
      gpu_origin_ = from.gpu_origin_;
    }
    if (cached_has_bits & 0x00000002u) {
      activation_ = from.activation_;
    }
    if (cached_has_bits & 0x00000004u) {
      output_layer_index_ = from.output_layer_index_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void TensorsToSegmentationCalculatorOptions::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.TensorsToSegmentationCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void TensorsToSegmentationCalculatorOptions::CopyFrom(const TensorsToSegmentationCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.TensorsToSegmentationCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool TensorsToSegmentationCalculatorOptions::IsInitialized() const {
  return true;
}

void TensorsToSegmentationCalculatorOptions::InternalSwap(TensorsToSegmentationCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(TensorsToSegmentationCalculatorOptions, activation_)
      + sizeof(TensorsToSegmentationCalculatorOptions::activation_)
      - PROTOBUF_FIELD_OFFSET(TensorsToSegmentationCalculatorOptions, gpu_origin_)>(
          reinterpret_cast<char*>(&gpu_origin_),
          reinterpret_cast<char*>(&other->gpu_origin_));
  swap(output_layer_index_, other->output_layer_index_);
}

::PROTOBUF_NAMESPACE_ID::Metadata TensorsToSegmentationCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5fsegmentation_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int TensorsToSegmentationCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::TensorsToSegmentationCalculatorOptions >, 11, false >
  TensorsToSegmentationCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::TensorsToSegmentationCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::TensorsToSegmentationCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::TensorsToSegmentationCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::TensorsToSegmentationCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <x/google/protobuf/port_undef.inc>
