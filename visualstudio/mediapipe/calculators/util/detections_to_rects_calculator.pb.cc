// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/util/detections_to_rects_calculator.proto

#include "mediapipe/calculators/util/detections_to_rects_calculator.pb.h"

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
constexpr DetectionsToRectsCalculatorOptions::DetectionsToRectsCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : rotation_vector_start_keypoint_index_(0)
  , rotation_vector_end_keypoint_index_(0)
  , rotation_vector_target_angle_(0)
  , rotation_vector_target_angle_degrees_(0)
  , output_zero_rect_for_empty_detections_(false)
  , conversion_mode_(0)
{}
struct DetectionsToRectsCalculatorOptionsDefaultTypeInternal {
  constexpr DetectionsToRectsCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~DetectionsToRectsCalculatorOptionsDefaultTypeInternal() {}
  union {
    DetectionsToRectsCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT DetectionsToRectsCalculatorOptionsDefaultTypeInternal _DetectionsToRectsCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto[1];
static const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* file_level_enum_descriptors_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionsToRectsCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionsToRectsCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionsToRectsCalculatorOptions, rotation_vector_start_keypoint_index_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionsToRectsCalculatorOptions, rotation_vector_end_keypoint_index_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionsToRectsCalculatorOptions, rotation_vector_target_angle_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionsToRectsCalculatorOptions, rotation_vector_target_angle_degrees_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionsToRectsCalculatorOptions, output_zero_rect_for_empty_detections_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionsToRectsCalculatorOptions, conversion_mode_),
  0,
  1,
  2,
  3,
  4,
  5,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 11, sizeof(::mediapipe::DetectionsToRectsCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_DetectionsToRectsCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\?mediapipe/calculators/util/detections_"
  "to_rects_calculator.proto\022\tmediapipe\032$me"
  "diapipe/framework/calculator.proto\"\375\003\n\"D"
  "etectionsToRectsCalculatorOptions\022,\n$rot"
  "ation_vector_start_keypoint_index\030\001 \001(\005\022"
  "*\n\"rotation_vector_end_keypoint_index\030\002 "
  "\001(\005\022$\n\034rotation_vector_target_angle\030\003 \001("
  "\002\022,\n$rotation_vector_target_angle_degree"
  "s\030\004 \001(\002\022-\n%output_zero_rect_for_empty_de"
  "tections\030\005 \001(\010\022U\n\017conversion_mode\030\006 \001(\0162"
  "<.mediapipe.DetectionsToRectsCalculatorO"
  "ptions.ConversionMode\"F\n\016ConversionMode\022"
  "\013\n\007DEFAULT\020\000\022\024\n\020USE_BOUNDING_BOX\020\001\022\021\n\rUS"
  "E_KEYPOINTS\020\0022[\n\003ext\022\034.mediapipe.Calcula"
  "torOptions\030\337\267\241} \001(\0132-.mediapipe.Detectio"
  "nsToRectsCalculatorOptions"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto_deps[1] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto = {
  false, false, 626, descriptor_table_protodef_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto, "mediapipe/calculators/util/detections_to_rects_calculator.proto", 
  &descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto_deps, 1, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto(&descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto);
namespace mediapipe {
const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* DetectionsToRectsCalculatorOptions_ConversionMode_descriptor() {
  ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto);
  return file_level_enum_descriptors_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto[0];
}
bool DetectionsToRectsCalculatorOptions_ConversionMode_IsValid(int value) {
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
constexpr DetectionsToRectsCalculatorOptions_ConversionMode DetectionsToRectsCalculatorOptions::DEFAULT;
constexpr DetectionsToRectsCalculatorOptions_ConversionMode DetectionsToRectsCalculatorOptions::USE_BOUNDING_BOX;
constexpr DetectionsToRectsCalculatorOptions_ConversionMode DetectionsToRectsCalculatorOptions::USE_KEYPOINTS;
constexpr DetectionsToRectsCalculatorOptions_ConversionMode DetectionsToRectsCalculatorOptions::ConversionMode_MIN;
constexpr DetectionsToRectsCalculatorOptions_ConversionMode DetectionsToRectsCalculatorOptions::ConversionMode_MAX;
constexpr int DetectionsToRectsCalculatorOptions::ConversionMode_ARRAYSIZE;
#endif  // (__cplusplus < 201703) && (!defined(_MSC_VER) || _MSC_VER >= 1900)

// ===================================================================

class DetectionsToRectsCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<DetectionsToRectsCalculatorOptions>()._has_bits_);
  static void set_has_rotation_vector_start_keypoint_index(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_rotation_vector_end_keypoint_index(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static void set_has_rotation_vector_target_angle(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static void set_has_rotation_vector_target_angle_degrees(HasBits* has_bits) {
    (*has_bits)[0] |= 8u;
  }
  static void set_has_output_zero_rect_for_empty_detections(HasBits* has_bits) {
    (*has_bits)[0] |= 16u;
  }
  static void set_has_conversion_mode(HasBits* has_bits) {
    (*has_bits)[0] |= 32u;
  }
};

DetectionsToRectsCalculatorOptions::DetectionsToRectsCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.DetectionsToRectsCalculatorOptions)
}
DetectionsToRectsCalculatorOptions::DetectionsToRectsCalculatorOptions(const DetectionsToRectsCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&rotation_vector_start_keypoint_index_, &from.rotation_vector_start_keypoint_index_,
    static_cast<size_t>(reinterpret_cast<char*>(&conversion_mode_) -
    reinterpret_cast<char*>(&rotation_vector_start_keypoint_index_)) + sizeof(conversion_mode_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.DetectionsToRectsCalculatorOptions)
}

void DetectionsToRectsCalculatorOptions::SharedCtor() {
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&rotation_vector_start_keypoint_index_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&conversion_mode_) -
    reinterpret_cast<char*>(&rotation_vector_start_keypoint_index_)) + sizeof(conversion_mode_));
}

DetectionsToRectsCalculatorOptions::~DetectionsToRectsCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.DetectionsToRectsCalculatorOptions)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void DetectionsToRectsCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void DetectionsToRectsCalculatorOptions::ArenaDtor(void* object) {
  DetectionsToRectsCalculatorOptions* _this = reinterpret_cast< DetectionsToRectsCalculatorOptions* >(object);
  (void)_this;
}
void DetectionsToRectsCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void DetectionsToRectsCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void DetectionsToRectsCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.DetectionsToRectsCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x0000003fu) {
    ::memset(&rotation_vector_start_keypoint_index_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&conversion_mode_) -
        reinterpret_cast<char*>(&rotation_vector_start_keypoint_index_)) + sizeof(conversion_mode_));
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* DetectionsToRectsCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional int32 rotation_vector_start_keypoint_index = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          _Internal::set_has_rotation_vector_start_keypoint_index(&has_bits);
          rotation_vector_start_keypoint_index_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 rotation_vector_end_keypoint_index = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          _Internal::set_has_rotation_vector_end_keypoint_index(&has_bits);
          rotation_vector_end_keypoint_index_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional float rotation_vector_target_angle = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 29)) {
          _Internal::set_has_rotation_vector_target_angle(&has_bits);
          rotation_vector_target_angle_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else goto handle_unusual;
        continue;
      // optional float rotation_vector_target_angle_degrees = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 37)) {
          _Internal::set_has_rotation_vector_target_angle_degrees(&has_bits);
          rotation_vector_target_angle_degrees_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else goto handle_unusual;
        continue;
      // optional bool output_zero_rect_for_empty_detections = 5;
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 40)) {
          _Internal::set_has_output_zero_rect_for_empty_detections(&has_bits);
          output_zero_rect_for_empty_detections_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional .mediapipe.DetectionsToRectsCalculatorOptions.ConversionMode conversion_mode = 6;
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 48)) {
          ::PROTOBUF_NAMESPACE_ID::uint64 val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
          if (PROTOBUF_PREDICT_TRUE(::mediapipe::DetectionsToRectsCalculatorOptions_ConversionMode_IsValid(val))) {
            _internal_set_conversion_mode(static_cast<::mediapipe::DetectionsToRectsCalculatorOptions_ConversionMode>(val));
          } else {
            ::PROTOBUF_NAMESPACE_ID::internal::WriteVarint(6, val, mutable_unknown_fields());
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

::PROTOBUF_NAMESPACE_ID::uint8* DetectionsToRectsCalculatorOptions::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.DetectionsToRectsCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional int32 rotation_vector_start_keypoint_index = 1;
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->_internal_rotation_vector_start_keypoint_index(), target);
  }

  // optional int32 rotation_vector_end_keypoint_index = 2;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(2, this->_internal_rotation_vector_end_keypoint_index(), target);
  }

  // optional float rotation_vector_target_angle = 3;
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(3, this->_internal_rotation_vector_target_angle(), target);
  }

  // optional float rotation_vector_target_angle_degrees = 4;
  if (cached_has_bits & 0x00000008u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(4, this->_internal_rotation_vector_target_angle_degrees(), target);
  }

  // optional bool output_zero_rect_for_empty_detections = 5;
  if (cached_has_bits & 0x00000010u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(5, this->_internal_output_zero_rect_for_empty_detections(), target);
  }

  // optional .mediapipe.DetectionsToRectsCalculatorOptions.ConversionMode conversion_mode = 6;
  if (cached_has_bits & 0x00000020u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteEnumToArray(
      6, this->_internal_conversion_mode(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.DetectionsToRectsCalculatorOptions)
  return target;
}

size_t DetectionsToRectsCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.DetectionsToRectsCalculatorOptions)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x0000003fu) {
    // optional int32 rotation_vector_start_keypoint_index = 1;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_rotation_vector_start_keypoint_index());
    }

    // optional int32 rotation_vector_end_keypoint_index = 2;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_rotation_vector_end_keypoint_index());
    }

    // optional float rotation_vector_target_angle = 3;
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 + 4;
    }

    // optional float rotation_vector_target_angle_degrees = 4;
    if (cached_has_bits & 0x00000008u) {
      total_size += 1 + 4;
    }

    // optional bool output_zero_rect_for_empty_detections = 5;
    if (cached_has_bits & 0x00000010u) {
      total_size += 1 + 1;
    }

    // optional .mediapipe.DetectionsToRectsCalculatorOptions.ConversionMode conversion_mode = 6;
    if (cached_has_bits & 0x00000020u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::EnumSize(this->_internal_conversion_mode());
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

void DetectionsToRectsCalculatorOptions::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.DetectionsToRectsCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  const DetectionsToRectsCalculatorOptions* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<DetectionsToRectsCalculatorOptions>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.DetectionsToRectsCalculatorOptions)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.DetectionsToRectsCalculatorOptions)
    MergeFrom(*source);
  }
}

void DetectionsToRectsCalculatorOptions::MergeFrom(const DetectionsToRectsCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.DetectionsToRectsCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x0000003fu) {
    if (cached_has_bits & 0x00000001u) {
      rotation_vector_start_keypoint_index_ = from.rotation_vector_start_keypoint_index_;
    }
    if (cached_has_bits & 0x00000002u) {
      rotation_vector_end_keypoint_index_ = from.rotation_vector_end_keypoint_index_;
    }
    if (cached_has_bits & 0x00000004u) {
      rotation_vector_target_angle_ = from.rotation_vector_target_angle_;
    }
    if (cached_has_bits & 0x00000008u) {
      rotation_vector_target_angle_degrees_ = from.rotation_vector_target_angle_degrees_;
    }
    if (cached_has_bits & 0x00000010u) {
      output_zero_rect_for_empty_detections_ = from.output_zero_rect_for_empty_detections_;
    }
    if (cached_has_bits & 0x00000020u) {
      conversion_mode_ = from.conversion_mode_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void DetectionsToRectsCalculatorOptions::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.DetectionsToRectsCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void DetectionsToRectsCalculatorOptions::CopyFrom(const DetectionsToRectsCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.DetectionsToRectsCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool DetectionsToRectsCalculatorOptions::IsInitialized() const {
  return true;
}

void DetectionsToRectsCalculatorOptions::InternalSwap(DetectionsToRectsCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(DetectionsToRectsCalculatorOptions, conversion_mode_)
      + sizeof(DetectionsToRectsCalculatorOptions::conversion_mode_)
      - PROTOBUF_FIELD_OFFSET(DetectionsToRectsCalculatorOptions, rotation_vector_start_keypoint_index_)>(
          reinterpret_cast<char*>(&rotation_vector_start_keypoint_index_),
          reinterpret_cast<char*>(&other->rotation_vector_start_keypoint_index_));
}

::PROTOBUF_NAMESPACE_ID::Metadata DetectionsToRectsCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int DetectionsToRectsCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::DetectionsToRectsCalculatorOptions >, 11, false >
  DetectionsToRectsCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::DetectionsToRectsCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::DetectionsToRectsCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::DetectionsToRectsCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::DetectionsToRectsCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <x/google/protobuf/port_undef.inc>
