// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/modules/objectron/calculators/frame_annotation_to_rect_calculator.proto

#include "mediapipe/modules/objectron/calculators/frame_annotation_to_rect_calculator.pb.h"

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
constexpr FrameAnnotationToRectCalculatorOptions::FrameAnnotationToRectCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : off_threshold_(40)
  , on_threshold_(41){}
struct FrameAnnotationToRectCalculatorOptionsDefaultTypeInternal {
  constexpr FrameAnnotationToRectCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~FrameAnnotationToRectCalculatorOptionsDefaultTypeInternal() {}
  union {
    FrameAnnotationToRectCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT FrameAnnotationToRectCalculatorOptionsDefaultTypeInternal _FrameAnnotationToRectCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::FrameAnnotationToRectCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FrameAnnotationToRectCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::FrameAnnotationToRectCalculatorOptions, off_threshold_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FrameAnnotationToRectCalculatorOptions, on_threshold_),
  0,
  1,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 7, sizeof(::mediapipe::FrameAnnotationToRectCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_FrameAnnotationToRectCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\nQmediapipe/modules/objectron/calculator"
  "s/frame_annotation_to_rect_calculator.pr"
  "oto\022\tmediapipe\032$mediapipe/framework/calc"
  "ulator.proto\"\277\001\n&FrameAnnotationToRectCa"
  "lculatorOptions\022\031\n\roff_threshold\030\001 \001(\002:\002"
  "40\022\030\n\014on_threshold\030\002 \001(\002:\002412`\n\003ext\022\034.me"
  "diapipe.CalculatorOptions\030\233\223\235\241\001 \001(\01321.me"
  "diapipe.FrameAnnotationToRectCalculatorO"
  "ptions"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto_deps[1] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto = {
  false, false, 326, descriptor_table_protodef_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto, "mediapipe/modules/objectron/calculators/frame_annotation_to_rect_calculator.proto", 
  &descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto_deps, 1, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto(&descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto);
namespace mediapipe {

// ===================================================================

class FrameAnnotationToRectCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<FrameAnnotationToRectCalculatorOptions>()._has_bits_);
  static void set_has_off_threshold(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_on_threshold(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
};

FrameAnnotationToRectCalculatorOptions::FrameAnnotationToRectCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.FrameAnnotationToRectCalculatorOptions)
}
FrameAnnotationToRectCalculatorOptions::FrameAnnotationToRectCalculatorOptions(const FrameAnnotationToRectCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&off_threshold_, &from.off_threshold_,
    static_cast<size_t>(reinterpret_cast<char*>(&on_threshold_) -
    reinterpret_cast<char*>(&off_threshold_)) + sizeof(on_threshold_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.FrameAnnotationToRectCalculatorOptions)
}

void FrameAnnotationToRectCalculatorOptions::SharedCtor() {
off_threshold_ = 40;
on_threshold_ = 41;
}

FrameAnnotationToRectCalculatorOptions::~FrameAnnotationToRectCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.FrameAnnotationToRectCalculatorOptions)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void FrameAnnotationToRectCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void FrameAnnotationToRectCalculatorOptions::ArenaDtor(void* object) {
  FrameAnnotationToRectCalculatorOptions* _this = reinterpret_cast< FrameAnnotationToRectCalculatorOptions* >(object);
  (void)_this;
}
void FrameAnnotationToRectCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void FrameAnnotationToRectCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void FrameAnnotationToRectCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.FrameAnnotationToRectCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    off_threshold_ = 40;
    on_threshold_ = 41;
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* FrameAnnotationToRectCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional float off_threshold = 1 [default = 40];
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 13)) {
          _Internal::set_has_off_threshold(&has_bits);
          off_threshold_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else goto handle_unusual;
        continue;
      // optional float on_threshold = 2 [default = 41];
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 21)) {
          _Internal::set_has_on_threshold(&has_bits);
          on_threshold_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
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

::PROTOBUF_NAMESPACE_ID::uint8* FrameAnnotationToRectCalculatorOptions::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.FrameAnnotationToRectCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional float off_threshold = 1 [default = 40];
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(1, this->_internal_off_threshold(), target);
  }

  // optional float on_threshold = 2 [default = 41];
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(2, this->_internal_on_threshold(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.FrameAnnotationToRectCalculatorOptions)
  return target;
}

size_t FrameAnnotationToRectCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.FrameAnnotationToRectCalculatorOptions)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    // optional float off_threshold = 1 [default = 40];
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 + 4;
    }

    // optional float on_threshold = 2 [default = 41];
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 + 4;
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

void FrameAnnotationToRectCalculatorOptions::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.FrameAnnotationToRectCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  const FrameAnnotationToRectCalculatorOptions* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<FrameAnnotationToRectCalculatorOptions>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.FrameAnnotationToRectCalculatorOptions)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.FrameAnnotationToRectCalculatorOptions)
    MergeFrom(*source);
  }
}

void FrameAnnotationToRectCalculatorOptions::MergeFrom(const FrameAnnotationToRectCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.FrameAnnotationToRectCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    if (cached_has_bits & 0x00000001u) {
      off_threshold_ = from.off_threshold_;
    }
    if (cached_has_bits & 0x00000002u) {
      on_threshold_ = from.on_threshold_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void FrameAnnotationToRectCalculatorOptions::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.FrameAnnotationToRectCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void FrameAnnotationToRectCalculatorOptions::CopyFrom(const FrameAnnotationToRectCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.FrameAnnotationToRectCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool FrameAnnotationToRectCalculatorOptions::IsInitialized() const {
  return true;
}

void FrameAnnotationToRectCalculatorOptions::InternalSwap(FrameAnnotationToRectCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  swap(off_threshold_, other->off_threshold_);
  swap(on_threshold_, other->on_threshold_);
}

::PROTOBUF_NAMESPACE_ID::Metadata FrameAnnotationToRectCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5fto_5frect_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int FrameAnnotationToRectCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::FrameAnnotationToRectCalculatorOptions >, 11, false >
  FrameAnnotationToRectCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::FrameAnnotationToRectCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::FrameAnnotationToRectCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::FrameAnnotationToRectCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::FrameAnnotationToRectCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <x/google/protobuf/port_undef.inc>
