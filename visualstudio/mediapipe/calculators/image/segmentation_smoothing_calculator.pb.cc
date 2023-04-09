// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/image/segmentation_smoothing_calculator.proto

#include "mediapipe/calculators/image/segmentation_smoothing_calculator.pb.h"

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
constexpr SegmentationSmoothingCalculatorOptions::SegmentationSmoothingCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : combine_with_previous_ratio_(0){}
struct SegmentationSmoothingCalculatorOptionsDefaultTypeInternal {
  constexpr SegmentationSmoothingCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~SegmentationSmoothingCalculatorOptionsDefaultTypeInternal() {}
  union {
    SegmentationSmoothingCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT SegmentationSmoothingCalculatorOptionsDefaultTypeInternal _SegmentationSmoothingCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::SegmentationSmoothingCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SegmentationSmoothingCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::SegmentationSmoothingCalculatorOptions, combine_with_previous_ratio_),
  0,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 6, sizeof(::mediapipe::SegmentationSmoothingCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_SegmentationSmoothingCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\nCmediapipe/calculators/image/segmentati"
  "on_smoothing_calculator.proto\022\tmediapipe"
  "\032$mediapipe/framework/calculator.proto\"\262"
  "\001\n&SegmentationSmoothingCalculatorOption"
  "s\022&\n\033combine_with_previous_ratio\030\001 \001(\002:\001"
  "02`\n\003ext\022\034.mediapipe.CalculatorOptions\030\350"
  "\231\374\263\001 \001(\01321.mediapipe.SegmentationSmoothi"
  "ngCalculatorOptions"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto_deps[1] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto = {
  false, false, 299, descriptor_table_protodef_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto, "mediapipe/calculators/image/segmentation_smoothing_calculator.proto", 
  &descriptor_table_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto_deps, 1, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto(&descriptor_table_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto);
namespace mediapipe {

// ===================================================================

class SegmentationSmoothingCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<SegmentationSmoothingCalculatorOptions>()._has_bits_);
  static void set_has_combine_with_previous_ratio(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
};

SegmentationSmoothingCalculatorOptions::SegmentationSmoothingCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.SegmentationSmoothingCalculatorOptions)
}
SegmentationSmoothingCalculatorOptions::SegmentationSmoothingCalculatorOptions(const SegmentationSmoothingCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  combine_with_previous_ratio_ = from.combine_with_previous_ratio_;
  // @@protoc_insertion_point(copy_constructor:mediapipe.SegmentationSmoothingCalculatorOptions)
}

void SegmentationSmoothingCalculatorOptions::SharedCtor() {
combine_with_previous_ratio_ = 0;
}

SegmentationSmoothingCalculatorOptions::~SegmentationSmoothingCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.SegmentationSmoothingCalculatorOptions)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void SegmentationSmoothingCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void SegmentationSmoothingCalculatorOptions::ArenaDtor(void* object) {
  SegmentationSmoothingCalculatorOptions* _this = reinterpret_cast< SegmentationSmoothingCalculatorOptions* >(object);
  (void)_this;
}
void SegmentationSmoothingCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void SegmentationSmoothingCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void SegmentationSmoothingCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.SegmentationSmoothingCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  combine_with_previous_ratio_ = 0;
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* SegmentationSmoothingCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional float combine_with_previous_ratio = 1 [default = 0];
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 13)) {
          _Internal::set_has_combine_with_previous_ratio(&has_bits);
          combine_with_previous_ratio_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
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

::PROTOBUF_NAMESPACE_ID::uint8* SegmentationSmoothingCalculatorOptions::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.SegmentationSmoothingCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional float combine_with_previous_ratio = 1 [default = 0];
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(1, this->_internal_combine_with_previous_ratio(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.SegmentationSmoothingCalculatorOptions)
  return target;
}

size_t SegmentationSmoothingCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.SegmentationSmoothingCalculatorOptions)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // optional float combine_with_previous_ratio = 1 [default = 0];
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000001u) {
    total_size += 1 + 4;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void SegmentationSmoothingCalculatorOptions::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.SegmentationSmoothingCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  const SegmentationSmoothingCalculatorOptions* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<SegmentationSmoothingCalculatorOptions>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.SegmentationSmoothingCalculatorOptions)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.SegmentationSmoothingCalculatorOptions)
    MergeFrom(*source);
  }
}

void SegmentationSmoothingCalculatorOptions::MergeFrom(const SegmentationSmoothingCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.SegmentationSmoothingCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from._internal_has_combine_with_previous_ratio()) {
    _internal_set_combine_with_previous_ratio(from._internal_combine_with_previous_ratio());
  }
}

void SegmentationSmoothingCalculatorOptions::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.SegmentationSmoothingCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void SegmentationSmoothingCalculatorOptions::CopyFrom(const SegmentationSmoothingCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.SegmentationSmoothingCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool SegmentationSmoothingCalculatorOptions::IsInitialized() const {
  return true;
}

void SegmentationSmoothingCalculatorOptions::InternalSwap(SegmentationSmoothingCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  swap(combine_with_previous_ratio_, other->combine_with_previous_ratio_);
}

::PROTOBUF_NAMESPACE_ID::Metadata SegmentationSmoothingCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2fimage_2fsegmentation_5fsmoothing_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int SegmentationSmoothingCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::SegmentationSmoothingCalculatorOptions >, 11, false >
  SegmentationSmoothingCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::SegmentationSmoothingCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::SegmentationSmoothingCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::SegmentationSmoothingCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::SegmentationSmoothingCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <x/google/protobuf/port_undef.inc>
