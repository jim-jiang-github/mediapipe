// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/util/detections_to_rects_calculator.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3015000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3015008 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/generated_enum_reflection.h>
#include <google/protobuf/unknown_field_set.h>
#include "mediapipe/framework/calculator.pb.h"
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[1]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto;
namespace mediapipe {
class DetectionsToRectsCalculatorOptions;
struct DetectionsToRectsCalculatorOptionsDefaultTypeInternal;
extern DetectionsToRectsCalculatorOptionsDefaultTypeInternal _DetectionsToRectsCalculatorOptions_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::DetectionsToRectsCalculatorOptions* Arena::CreateMaybeMessage<::mediapipe::DetectionsToRectsCalculatorOptions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

enum DetectionsToRectsCalculatorOptions_ConversionMode : int {
  DetectionsToRectsCalculatorOptions_ConversionMode_DEFAULT = 0,
  DetectionsToRectsCalculatorOptions_ConversionMode_USE_BOUNDING_BOX = 1,
  DetectionsToRectsCalculatorOptions_ConversionMode_USE_KEYPOINTS = 2
};
bool DetectionsToRectsCalculatorOptions_ConversionMode_IsValid(int value);
constexpr DetectionsToRectsCalculatorOptions_ConversionMode DetectionsToRectsCalculatorOptions_ConversionMode_ConversionMode_MIN = DetectionsToRectsCalculatorOptions_ConversionMode_DEFAULT;
constexpr DetectionsToRectsCalculatorOptions_ConversionMode DetectionsToRectsCalculatorOptions_ConversionMode_ConversionMode_MAX = DetectionsToRectsCalculatorOptions_ConversionMode_USE_KEYPOINTS;
constexpr int DetectionsToRectsCalculatorOptions_ConversionMode_ConversionMode_ARRAYSIZE = DetectionsToRectsCalculatorOptions_ConversionMode_ConversionMode_MAX + 1;

const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* DetectionsToRectsCalculatorOptions_ConversionMode_descriptor();
template<typename T>
inline const std::string& DetectionsToRectsCalculatorOptions_ConversionMode_Name(T enum_t_value) {
  static_assert(::std::is_same<T, DetectionsToRectsCalculatorOptions_ConversionMode>::value ||
    ::std::is_integral<T>::value,
    "Incorrect type passed to function DetectionsToRectsCalculatorOptions_ConversionMode_Name.");
  return ::PROTOBUF_NAMESPACE_ID::internal::NameOfEnum(
    DetectionsToRectsCalculatorOptions_ConversionMode_descriptor(), enum_t_value);
}
inline bool DetectionsToRectsCalculatorOptions_ConversionMode_Parse(
    ::PROTOBUF_NAMESPACE_ID::ConstStringParam name, DetectionsToRectsCalculatorOptions_ConversionMode* value) {
  return ::PROTOBUF_NAMESPACE_ID::internal::ParseNamedEnum<DetectionsToRectsCalculatorOptions_ConversionMode>(
    DetectionsToRectsCalculatorOptions_ConversionMode_descriptor(), name, value);
}
// ===================================================================

class DetectionsToRectsCalculatorOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.DetectionsToRectsCalculatorOptions) */ {
 public:
  inline DetectionsToRectsCalculatorOptions() : DetectionsToRectsCalculatorOptions(nullptr) {}
  ~DetectionsToRectsCalculatorOptions() override;
  explicit constexpr DetectionsToRectsCalculatorOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  DetectionsToRectsCalculatorOptions(const DetectionsToRectsCalculatorOptions& from);
  DetectionsToRectsCalculatorOptions(DetectionsToRectsCalculatorOptions&& from) noexcept
    : DetectionsToRectsCalculatorOptions() {
    *this = ::std::move(from);
  }

  inline DetectionsToRectsCalculatorOptions& operator=(const DetectionsToRectsCalculatorOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline DetectionsToRectsCalculatorOptions& operator=(DetectionsToRectsCalculatorOptions&& from) noexcept {
    if (GetArena() == from.GetArena()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance);
  }
  inline ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const DetectionsToRectsCalculatorOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const DetectionsToRectsCalculatorOptions* internal_default_instance() {
    return reinterpret_cast<const DetectionsToRectsCalculatorOptions*>(
               &_DetectionsToRectsCalculatorOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(DetectionsToRectsCalculatorOptions& a, DetectionsToRectsCalculatorOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(DetectionsToRectsCalculatorOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(DetectionsToRectsCalculatorOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline DetectionsToRectsCalculatorOptions* New() const final {
    return CreateMaybeMessage<DetectionsToRectsCalculatorOptions>(nullptr);
  }

  DetectionsToRectsCalculatorOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<DetectionsToRectsCalculatorOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const DetectionsToRectsCalculatorOptions& from);
  void MergeFrom(const DetectionsToRectsCalculatorOptions& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(DetectionsToRectsCalculatorOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.DetectionsToRectsCalculatorOptions";
  }
  protected:
  explicit DetectionsToRectsCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef DetectionsToRectsCalculatorOptions_ConversionMode ConversionMode;
  static constexpr ConversionMode DEFAULT =
    DetectionsToRectsCalculatorOptions_ConversionMode_DEFAULT;
  static constexpr ConversionMode USE_BOUNDING_BOX =
    DetectionsToRectsCalculatorOptions_ConversionMode_USE_BOUNDING_BOX;
  static constexpr ConversionMode USE_KEYPOINTS =
    DetectionsToRectsCalculatorOptions_ConversionMode_USE_KEYPOINTS;
  static inline bool ConversionMode_IsValid(int value) {
    return DetectionsToRectsCalculatorOptions_ConversionMode_IsValid(value);
  }
  static constexpr ConversionMode ConversionMode_MIN =
    DetectionsToRectsCalculatorOptions_ConversionMode_ConversionMode_MIN;
  static constexpr ConversionMode ConversionMode_MAX =
    DetectionsToRectsCalculatorOptions_ConversionMode_ConversionMode_MAX;
  static constexpr int ConversionMode_ARRAYSIZE =
    DetectionsToRectsCalculatorOptions_ConversionMode_ConversionMode_ARRAYSIZE;
  static inline const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor*
  ConversionMode_descriptor() {
    return DetectionsToRectsCalculatorOptions_ConversionMode_descriptor();
  }
  template<typename T>
  static inline const std::string& ConversionMode_Name(T enum_t_value) {
    static_assert(::std::is_same<T, ConversionMode>::value ||
      ::std::is_integral<T>::value,
      "Incorrect type passed to function ConversionMode_Name.");
    return DetectionsToRectsCalculatorOptions_ConversionMode_Name(enum_t_value);
  }
  static inline bool ConversionMode_Parse(::PROTOBUF_NAMESPACE_ID::ConstStringParam name,
      ConversionMode* value) {
    return DetectionsToRectsCalculatorOptions_ConversionMode_Parse(name, value);
  }

  // accessors -------------------------------------------------------

  enum : int {
    kRotationVectorStartKeypointIndexFieldNumber = 1,
    kRotationVectorEndKeypointIndexFieldNumber = 2,
    kRotationVectorTargetAngleFieldNumber = 3,
    kRotationVectorTargetAngleDegreesFieldNumber = 4,
    kOutputZeroRectForEmptyDetectionsFieldNumber = 5,
    kConversionModeFieldNumber = 6,
  };
  // optional int32 rotation_vector_start_keypoint_index = 1;
  bool has_rotation_vector_start_keypoint_index() const;
  private:
  bool _internal_has_rotation_vector_start_keypoint_index() const;
  public:
  void clear_rotation_vector_start_keypoint_index();
  ::PROTOBUF_NAMESPACE_ID::int32 rotation_vector_start_keypoint_index() const;
  void set_rotation_vector_start_keypoint_index(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_rotation_vector_start_keypoint_index() const;
  void _internal_set_rotation_vector_start_keypoint_index(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional int32 rotation_vector_end_keypoint_index = 2;
  bool has_rotation_vector_end_keypoint_index() const;
  private:
  bool _internal_has_rotation_vector_end_keypoint_index() const;
  public:
  void clear_rotation_vector_end_keypoint_index();
  ::PROTOBUF_NAMESPACE_ID::int32 rotation_vector_end_keypoint_index() const;
  void set_rotation_vector_end_keypoint_index(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_rotation_vector_end_keypoint_index() const;
  void _internal_set_rotation_vector_end_keypoint_index(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional float rotation_vector_target_angle = 3;
  bool has_rotation_vector_target_angle() const;
  private:
  bool _internal_has_rotation_vector_target_angle() const;
  public:
  void clear_rotation_vector_target_angle();
  float rotation_vector_target_angle() const;
  void set_rotation_vector_target_angle(float value);
  private:
  float _internal_rotation_vector_target_angle() const;
  void _internal_set_rotation_vector_target_angle(float value);
  public:

  // optional float rotation_vector_target_angle_degrees = 4;
  bool has_rotation_vector_target_angle_degrees() const;
  private:
  bool _internal_has_rotation_vector_target_angle_degrees() const;
  public:
  void clear_rotation_vector_target_angle_degrees();
  float rotation_vector_target_angle_degrees() const;
  void set_rotation_vector_target_angle_degrees(float value);
  private:
  float _internal_rotation_vector_target_angle_degrees() const;
  void _internal_set_rotation_vector_target_angle_degrees(float value);
  public:

  // optional bool output_zero_rect_for_empty_detections = 5;
  bool has_output_zero_rect_for_empty_detections() const;
  private:
  bool _internal_has_output_zero_rect_for_empty_detections() const;
  public:
  void clear_output_zero_rect_for_empty_detections();
  bool output_zero_rect_for_empty_detections() const;
  void set_output_zero_rect_for_empty_detections(bool value);
  private:
  bool _internal_output_zero_rect_for_empty_detections() const;
  void _internal_set_output_zero_rect_for_empty_detections(bool value);
  public:

  // optional .mediapipe.DetectionsToRectsCalculatorOptions.ConversionMode conversion_mode = 6;
  bool has_conversion_mode() const;
  private:
  bool _internal_has_conversion_mode() const;
  public:
  void clear_conversion_mode();
  ::mediapipe::DetectionsToRectsCalculatorOptions_ConversionMode conversion_mode() const;
  void set_conversion_mode(::mediapipe::DetectionsToRectsCalculatorOptions_ConversionMode value);
  private:
  ::mediapipe::DetectionsToRectsCalculatorOptions_ConversionMode _internal_conversion_mode() const;
  void _internal_set_conversion_mode(::mediapipe::DetectionsToRectsCalculatorOptions_ConversionMode value);
  public:

  static const int kExtFieldNumber = 262691807;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::DetectionsToRectsCalculatorOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.DetectionsToRectsCalculatorOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::int32 rotation_vector_start_keypoint_index_;
  ::PROTOBUF_NAMESPACE_ID::int32 rotation_vector_end_keypoint_index_;
  float rotation_vector_target_angle_;
  float rotation_vector_target_angle_degrees_;
  bool output_zero_rect_for_empty_detections_;
  int conversion_mode_;
  friend struct ::TableStruct_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// DetectionsToRectsCalculatorOptions

// optional int32 rotation_vector_start_keypoint_index = 1;
inline bool DetectionsToRectsCalculatorOptions::_internal_has_rotation_vector_start_keypoint_index() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool DetectionsToRectsCalculatorOptions::has_rotation_vector_start_keypoint_index() const {
  return _internal_has_rotation_vector_start_keypoint_index();
}
inline void DetectionsToRectsCalculatorOptions::clear_rotation_vector_start_keypoint_index() {
  rotation_vector_start_keypoint_index_ = 0;
  _has_bits_[0] &= ~0x00000001u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 DetectionsToRectsCalculatorOptions::_internal_rotation_vector_start_keypoint_index() const {
  return rotation_vector_start_keypoint_index_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 DetectionsToRectsCalculatorOptions::rotation_vector_start_keypoint_index() const {
  // @@protoc_insertion_point(field_get:mediapipe.DetectionsToRectsCalculatorOptions.rotation_vector_start_keypoint_index)
  return _internal_rotation_vector_start_keypoint_index();
}
inline void DetectionsToRectsCalculatorOptions::_internal_set_rotation_vector_start_keypoint_index(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000001u;
  rotation_vector_start_keypoint_index_ = value;
}
inline void DetectionsToRectsCalculatorOptions::set_rotation_vector_start_keypoint_index(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_rotation_vector_start_keypoint_index(value);
  // @@protoc_insertion_point(field_set:mediapipe.DetectionsToRectsCalculatorOptions.rotation_vector_start_keypoint_index)
}

// optional int32 rotation_vector_end_keypoint_index = 2;
inline bool DetectionsToRectsCalculatorOptions::_internal_has_rotation_vector_end_keypoint_index() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool DetectionsToRectsCalculatorOptions::has_rotation_vector_end_keypoint_index() const {
  return _internal_has_rotation_vector_end_keypoint_index();
}
inline void DetectionsToRectsCalculatorOptions::clear_rotation_vector_end_keypoint_index() {
  rotation_vector_end_keypoint_index_ = 0;
  _has_bits_[0] &= ~0x00000002u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 DetectionsToRectsCalculatorOptions::_internal_rotation_vector_end_keypoint_index() const {
  return rotation_vector_end_keypoint_index_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 DetectionsToRectsCalculatorOptions::rotation_vector_end_keypoint_index() const {
  // @@protoc_insertion_point(field_get:mediapipe.DetectionsToRectsCalculatorOptions.rotation_vector_end_keypoint_index)
  return _internal_rotation_vector_end_keypoint_index();
}
inline void DetectionsToRectsCalculatorOptions::_internal_set_rotation_vector_end_keypoint_index(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000002u;
  rotation_vector_end_keypoint_index_ = value;
}
inline void DetectionsToRectsCalculatorOptions::set_rotation_vector_end_keypoint_index(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_rotation_vector_end_keypoint_index(value);
  // @@protoc_insertion_point(field_set:mediapipe.DetectionsToRectsCalculatorOptions.rotation_vector_end_keypoint_index)
}

// optional float rotation_vector_target_angle = 3;
inline bool DetectionsToRectsCalculatorOptions::_internal_has_rotation_vector_target_angle() const {
  bool value = (_has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool DetectionsToRectsCalculatorOptions::has_rotation_vector_target_angle() const {
  return _internal_has_rotation_vector_target_angle();
}
inline void DetectionsToRectsCalculatorOptions::clear_rotation_vector_target_angle() {
  rotation_vector_target_angle_ = 0;
  _has_bits_[0] &= ~0x00000004u;
}
inline float DetectionsToRectsCalculatorOptions::_internal_rotation_vector_target_angle() const {
  return rotation_vector_target_angle_;
}
inline float DetectionsToRectsCalculatorOptions::rotation_vector_target_angle() const {
  // @@protoc_insertion_point(field_get:mediapipe.DetectionsToRectsCalculatorOptions.rotation_vector_target_angle)
  return _internal_rotation_vector_target_angle();
}
inline void DetectionsToRectsCalculatorOptions::_internal_set_rotation_vector_target_angle(float value) {
  _has_bits_[0] |= 0x00000004u;
  rotation_vector_target_angle_ = value;
}
inline void DetectionsToRectsCalculatorOptions::set_rotation_vector_target_angle(float value) {
  _internal_set_rotation_vector_target_angle(value);
  // @@protoc_insertion_point(field_set:mediapipe.DetectionsToRectsCalculatorOptions.rotation_vector_target_angle)
}

// optional float rotation_vector_target_angle_degrees = 4;
inline bool DetectionsToRectsCalculatorOptions::_internal_has_rotation_vector_target_angle_degrees() const {
  bool value = (_has_bits_[0] & 0x00000008u) != 0;
  return value;
}
inline bool DetectionsToRectsCalculatorOptions::has_rotation_vector_target_angle_degrees() const {
  return _internal_has_rotation_vector_target_angle_degrees();
}
inline void DetectionsToRectsCalculatorOptions::clear_rotation_vector_target_angle_degrees() {
  rotation_vector_target_angle_degrees_ = 0;
  _has_bits_[0] &= ~0x00000008u;
}
inline float DetectionsToRectsCalculatorOptions::_internal_rotation_vector_target_angle_degrees() const {
  return rotation_vector_target_angle_degrees_;
}
inline float DetectionsToRectsCalculatorOptions::rotation_vector_target_angle_degrees() const {
  // @@protoc_insertion_point(field_get:mediapipe.DetectionsToRectsCalculatorOptions.rotation_vector_target_angle_degrees)
  return _internal_rotation_vector_target_angle_degrees();
}
inline void DetectionsToRectsCalculatorOptions::_internal_set_rotation_vector_target_angle_degrees(float value) {
  _has_bits_[0] |= 0x00000008u;
  rotation_vector_target_angle_degrees_ = value;
}
inline void DetectionsToRectsCalculatorOptions::set_rotation_vector_target_angle_degrees(float value) {
  _internal_set_rotation_vector_target_angle_degrees(value);
  // @@protoc_insertion_point(field_set:mediapipe.DetectionsToRectsCalculatorOptions.rotation_vector_target_angle_degrees)
}

// optional bool output_zero_rect_for_empty_detections = 5;
inline bool DetectionsToRectsCalculatorOptions::_internal_has_output_zero_rect_for_empty_detections() const {
  bool value = (_has_bits_[0] & 0x00000010u) != 0;
  return value;
}
inline bool DetectionsToRectsCalculatorOptions::has_output_zero_rect_for_empty_detections() const {
  return _internal_has_output_zero_rect_for_empty_detections();
}
inline void DetectionsToRectsCalculatorOptions::clear_output_zero_rect_for_empty_detections() {
  output_zero_rect_for_empty_detections_ = false;
  _has_bits_[0] &= ~0x00000010u;
}
inline bool DetectionsToRectsCalculatorOptions::_internal_output_zero_rect_for_empty_detections() const {
  return output_zero_rect_for_empty_detections_;
}
inline bool DetectionsToRectsCalculatorOptions::output_zero_rect_for_empty_detections() const {
  // @@protoc_insertion_point(field_get:mediapipe.DetectionsToRectsCalculatorOptions.output_zero_rect_for_empty_detections)
  return _internal_output_zero_rect_for_empty_detections();
}
inline void DetectionsToRectsCalculatorOptions::_internal_set_output_zero_rect_for_empty_detections(bool value) {
  _has_bits_[0] |= 0x00000010u;
  output_zero_rect_for_empty_detections_ = value;
}
inline void DetectionsToRectsCalculatorOptions::set_output_zero_rect_for_empty_detections(bool value) {
  _internal_set_output_zero_rect_for_empty_detections(value);
  // @@protoc_insertion_point(field_set:mediapipe.DetectionsToRectsCalculatorOptions.output_zero_rect_for_empty_detections)
}

// optional .mediapipe.DetectionsToRectsCalculatorOptions.ConversionMode conversion_mode = 6;
inline bool DetectionsToRectsCalculatorOptions::_internal_has_conversion_mode() const {
  bool value = (_has_bits_[0] & 0x00000020u) != 0;
  return value;
}
inline bool DetectionsToRectsCalculatorOptions::has_conversion_mode() const {
  return _internal_has_conversion_mode();
}
inline void DetectionsToRectsCalculatorOptions::clear_conversion_mode() {
  conversion_mode_ = 0;
  _has_bits_[0] &= ~0x00000020u;
}
inline ::mediapipe::DetectionsToRectsCalculatorOptions_ConversionMode DetectionsToRectsCalculatorOptions::_internal_conversion_mode() const {
  return static_cast< ::mediapipe::DetectionsToRectsCalculatorOptions_ConversionMode >(conversion_mode_);
}
inline ::mediapipe::DetectionsToRectsCalculatorOptions_ConversionMode DetectionsToRectsCalculatorOptions::conversion_mode() const {
  // @@protoc_insertion_point(field_get:mediapipe.DetectionsToRectsCalculatorOptions.conversion_mode)
  return _internal_conversion_mode();
}
inline void DetectionsToRectsCalculatorOptions::_internal_set_conversion_mode(::mediapipe::DetectionsToRectsCalculatorOptions_ConversionMode value) {
  assert(::mediapipe::DetectionsToRectsCalculatorOptions_ConversionMode_IsValid(value));
  _has_bits_[0] |= 0x00000020u;
  conversion_mode_ = value;
}
inline void DetectionsToRectsCalculatorOptions::set_conversion_mode(::mediapipe::DetectionsToRectsCalculatorOptions_ConversionMode value) {
  _internal_set_conversion_mode(value);
  // @@protoc_insertion_point(field_set:mediapipe.DetectionsToRectsCalculatorOptions.conversion_mode)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

PROTOBUF_NAMESPACE_OPEN

template <> struct is_proto_enum< ::mediapipe::DetectionsToRectsCalculatorOptions_ConversionMode> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::mediapipe::DetectionsToRectsCalculatorOptions_ConversionMode>() {
  return ::mediapipe::DetectionsToRectsCalculatorOptions_ConversionMode_descriptor();
}

PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frects_5fcalculator_2eproto
