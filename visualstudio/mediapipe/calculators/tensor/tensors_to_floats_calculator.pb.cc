// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/tensor/tensors_to_floats_calculator.proto

#include "mediapipe/calculators/tensor/tensors_to_floats_calculator.pb.h"

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
constexpr TensorsToFloatsCalculatorOptions::TensorsToFloatsCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : activation_(0)
{}
struct TensorsToFloatsCalculatorOptionsDefaultTypeInternal {
  constexpr TensorsToFloatsCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~TensorsToFloatsCalculatorOptionsDefaultTypeInternal() {}
  union {
    TensorsToFloatsCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT TensorsToFloatsCalculatorOptionsDefaultTypeInternal _TensorsToFloatsCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto[1];
static const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* file_level_enum_descriptors_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::TensorsToFloatsCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TensorsToFloatsCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::TensorsToFloatsCalculatorOptions, activation_),
  0,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 6, sizeof(::mediapipe::TensorsToFloatsCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_TensorsToFloatsCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\?mediapipe/calculators/tensor/tensors_t"
  "o_floats_calculator.proto\022\tmediapipe\032$me"
  "diapipe/framework/calculator.proto\"\365\001\n T"
  "ensorsToFloatsCalculatorOptions\022P\n\nactiv"
  "ation\030\001 \001(\01626.mediapipe.TensorsToFloatsC"
  "alculatorOptions.Activation:\004NONE\"#\n\nAct"
  "ivation\022\010\n\004NONE\020\000\022\013\n\007SIGMOID\020\0012Z\n\003ext\022\034."
  "mediapipe.CalculatorOptions\030\353\302\345\243\001 \001(\0132+."
  "mediapipe.TensorsToFloatsCalculatorOptio"
  "ns"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto_deps[1] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto = {
  false, false, 362, descriptor_table_protodef_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto, "mediapipe/calculators/tensor/tensors_to_floats_calculator.proto", 
  &descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto_deps, 1, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto(&descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto);
namespace mediapipe {
const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* TensorsToFloatsCalculatorOptions_Activation_descriptor() {
  ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto);
  return file_level_enum_descriptors_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto[0];
}
bool TensorsToFloatsCalculatorOptions_Activation_IsValid(int value) {
  switch (value) {
    case 0:
    case 1:
      return true;
    default:
      return false;
  }
}

#if (__cplusplus < 201703) && (!defined(_MSC_VER) || _MSC_VER >= 1900)
constexpr TensorsToFloatsCalculatorOptions_Activation TensorsToFloatsCalculatorOptions::NONE;
constexpr TensorsToFloatsCalculatorOptions_Activation TensorsToFloatsCalculatorOptions::SIGMOID;
constexpr TensorsToFloatsCalculatorOptions_Activation TensorsToFloatsCalculatorOptions::Activation_MIN;
constexpr TensorsToFloatsCalculatorOptions_Activation TensorsToFloatsCalculatorOptions::Activation_MAX;
constexpr int TensorsToFloatsCalculatorOptions::Activation_ARRAYSIZE;
#endif  // (__cplusplus < 201703) && (!defined(_MSC_VER) || _MSC_VER >= 1900)

// ===================================================================

class TensorsToFloatsCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<TensorsToFloatsCalculatorOptions>()._has_bits_);
  static void set_has_activation(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
};

TensorsToFloatsCalculatorOptions::TensorsToFloatsCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.TensorsToFloatsCalculatorOptions)
}
TensorsToFloatsCalculatorOptions::TensorsToFloatsCalculatorOptions(const TensorsToFloatsCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  activation_ = from.activation_;
  // @@protoc_insertion_point(copy_constructor:mediapipe.TensorsToFloatsCalculatorOptions)
}

void TensorsToFloatsCalculatorOptions::SharedCtor() {
activation_ = 0;
}

TensorsToFloatsCalculatorOptions::~TensorsToFloatsCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.TensorsToFloatsCalculatorOptions)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void TensorsToFloatsCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void TensorsToFloatsCalculatorOptions::ArenaDtor(void* object) {
  TensorsToFloatsCalculatorOptions* _this = reinterpret_cast< TensorsToFloatsCalculatorOptions* >(object);
  (void)_this;
}
void TensorsToFloatsCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void TensorsToFloatsCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void TensorsToFloatsCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.TensorsToFloatsCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  activation_ = 0;
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* TensorsToFloatsCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional .mediapipe.TensorsToFloatsCalculatorOptions.Activation activation = 1 [default = NONE];
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          ::PROTOBUF_NAMESPACE_ID::uint64 val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
          if (PROTOBUF_PREDICT_TRUE(::mediapipe::TensorsToFloatsCalculatorOptions_Activation_IsValid(val))) {
            _internal_set_activation(static_cast<::mediapipe::TensorsToFloatsCalculatorOptions_Activation>(val));
          } else {
            ::PROTOBUF_NAMESPACE_ID::internal::WriteVarint(1, val, mutable_unknown_fields());
          }
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

::PROTOBUF_NAMESPACE_ID::uint8* TensorsToFloatsCalculatorOptions::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.TensorsToFloatsCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional .mediapipe.TensorsToFloatsCalculatorOptions.Activation activation = 1 [default = NONE];
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteEnumToArray(
      1, this->_internal_activation(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.TensorsToFloatsCalculatorOptions)
  return target;
}

size_t TensorsToFloatsCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.TensorsToFloatsCalculatorOptions)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // optional .mediapipe.TensorsToFloatsCalculatorOptions.Activation activation = 1 [default = NONE];
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000001u) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::EnumSize(this->_internal_activation());
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void TensorsToFloatsCalculatorOptions::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.TensorsToFloatsCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  const TensorsToFloatsCalculatorOptions* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<TensorsToFloatsCalculatorOptions>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.TensorsToFloatsCalculatorOptions)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.TensorsToFloatsCalculatorOptions)
    MergeFrom(*source);
  }
}

void TensorsToFloatsCalculatorOptions::MergeFrom(const TensorsToFloatsCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.TensorsToFloatsCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from._internal_has_activation()) {
    _internal_set_activation(from._internal_activation());
  }
}

void TensorsToFloatsCalculatorOptions::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.TensorsToFloatsCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void TensorsToFloatsCalculatorOptions::CopyFrom(const TensorsToFloatsCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.TensorsToFloatsCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool TensorsToFloatsCalculatorOptions::IsInitialized() const {
  return true;
}

void TensorsToFloatsCalculatorOptions::InternalSwap(TensorsToFloatsCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  swap(activation_, other->activation_);
}

::PROTOBUF_NAMESPACE_ID::Metadata TensorsToFloatsCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2ftensor_2ftensors_5fto_5ffloats_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int TensorsToFloatsCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::TensorsToFloatsCalculatorOptions >, 11, false >
  TensorsToFloatsCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::TensorsToFloatsCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::TensorsToFloatsCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::TensorsToFloatsCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::TensorsToFloatsCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <x/google/protobuf/port_undef.inc>
