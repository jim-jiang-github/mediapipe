// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/modules/objectron/calculators/frame_annotation_tracker_calculator.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5ftracker_5fcalculator_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5ftracker_5fcalculator_2eproto

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
#include <google/protobuf/unknown_field_set.h>
#include "mediapipe/framework/calculator.pb.h"
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5ftracker_5fcalculator_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5ftracker_5fcalculator_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5ftracker_5fcalculator_2eproto;
namespace mediapipe {
class FrameAnnotationTrackerCalculatorOptions;
struct FrameAnnotationTrackerCalculatorOptionsDefaultTypeInternal;
extern FrameAnnotationTrackerCalculatorOptionsDefaultTypeInternal _FrameAnnotationTrackerCalculatorOptions_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::FrameAnnotationTrackerCalculatorOptions* Arena::CreateMaybeMessage<::mediapipe::FrameAnnotationTrackerCalculatorOptions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class FrameAnnotationTrackerCalculatorOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.FrameAnnotationTrackerCalculatorOptions) */ {
 public:
  inline FrameAnnotationTrackerCalculatorOptions() : FrameAnnotationTrackerCalculatorOptions(nullptr) {}
  ~FrameAnnotationTrackerCalculatorOptions() override;
  explicit constexpr FrameAnnotationTrackerCalculatorOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  FrameAnnotationTrackerCalculatorOptions(const FrameAnnotationTrackerCalculatorOptions& from);
  FrameAnnotationTrackerCalculatorOptions(FrameAnnotationTrackerCalculatorOptions&& from) noexcept
    : FrameAnnotationTrackerCalculatorOptions() {
    *this = ::std::move(from);
  }

  inline FrameAnnotationTrackerCalculatorOptions& operator=(const FrameAnnotationTrackerCalculatorOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline FrameAnnotationTrackerCalculatorOptions& operator=(FrameAnnotationTrackerCalculatorOptions&& from) noexcept {
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
  static const FrameAnnotationTrackerCalculatorOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const FrameAnnotationTrackerCalculatorOptions* internal_default_instance() {
    return reinterpret_cast<const FrameAnnotationTrackerCalculatorOptions*>(
               &_FrameAnnotationTrackerCalculatorOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(FrameAnnotationTrackerCalculatorOptions& a, FrameAnnotationTrackerCalculatorOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(FrameAnnotationTrackerCalculatorOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(FrameAnnotationTrackerCalculatorOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline FrameAnnotationTrackerCalculatorOptions* New() const final {
    return CreateMaybeMessage<FrameAnnotationTrackerCalculatorOptions>(nullptr);
  }

  FrameAnnotationTrackerCalculatorOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<FrameAnnotationTrackerCalculatorOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const FrameAnnotationTrackerCalculatorOptions& from);
  void MergeFrom(const FrameAnnotationTrackerCalculatorOptions& from);
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
  void InternalSwap(FrameAnnotationTrackerCalculatorOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.FrameAnnotationTrackerCalculatorOptions";
  }
  protected:
  explicit FrameAnnotationTrackerCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kImgWidthFieldNumber = 2,
    kImgHeightFieldNumber = 3,
    kIouThresholdFieldNumber = 1,
  };
  // optional float img_width = 2;
  bool has_img_width() const;
  private:
  bool _internal_has_img_width() const;
  public:
  void clear_img_width();
  float img_width() const;
  void set_img_width(float value);
  private:
  float _internal_img_width() const;
  void _internal_set_img_width(float value);
  public:

  // optional float img_height = 3;
  bool has_img_height() const;
  private:
  bool _internal_has_img_height() const;
  public:
  void clear_img_height();
  float img_height() const;
  void set_img_height(float value);
  private:
  float _internal_img_height() const;
  void _internal_set_img_height(float value);
  public:

  // optional float iou_threshold = 1 [default = 0.5];
  bool has_iou_threshold() const;
  private:
  bool _internal_has_iou_threshold() const;
  public:
  void clear_iou_threshold();
  float iou_threshold() const;
  void set_iou_threshold(float value);
  private:
  float _internal_iou_threshold() const;
  void _internal_set_iou_threshold(float value);
  public:

  static const int kExtFieldNumber = 291291253;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::FrameAnnotationTrackerCalculatorOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.FrameAnnotationTrackerCalculatorOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  float img_width_;
  float img_height_;
  float iou_threshold_;
  friend struct ::TableStruct_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5ftracker_5fcalculator_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// FrameAnnotationTrackerCalculatorOptions

// optional float iou_threshold = 1 [default = 0.5];
inline bool FrameAnnotationTrackerCalculatorOptions::_internal_has_iou_threshold() const {
  bool value = (_has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool FrameAnnotationTrackerCalculatorOptions::has_iou_threshold() const {
  return _internal_has_iou_threshold();
}
inline void FrameAnnotationTrackerCalculatorOptions::clear_iou_threshold() {
  iou_threshold_ = 0.5f;
  _has_bits_[0] &= ~0x00000004u;
}
inline float FrameAnnotationTrackerCalculatorOptions::_internal_iou_threshold() const {
  return iou_threshold_;
}
inline float FrameAnnotationTrackerCalculatorOptions::iou_threshold() const {
  // @@protoc_insertion_point(field_get:mediapipe.FrameAnnotationTrackerCalculatorOptions.iou_threshold)
  return _internal_iou_threshold();
}
inline void FrameAnnotationTrackerCalculatorOptions::_internal_set_iou_threshold(float value) {
  _has_bits_[0] |= 0x00000004u;
  iou_threshold_ = value;
}
inline void FrameAnnotationTrackerCalculatorOptions::set_iou_threshold(float value) {
  _internal_set_iou_threshold(value);
  // @@protoc_insertion_point(field_set:mediapipe.FrameAnnotationTrackerCalculatorOptions.iou_threshold)
}

// optional float img_width = 2;
inline bool FrameAnnotationTrackerCalculatorOptions::_internal_has_img_width() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool FrameAnnotationTrackerCalculatorOptions::has_img_width() const {
  return _internal_has_img_width();
}
inline void FrameAnnotationTrackerCalculatorOptions::clear_img_width() {
  img_width_ = 0;
  _has_bits_[0] &= ~0x00000001u;
}
inline float FrameAnnotationTrackerCalculatorOptions::_internal_img_width() const {
  return img_width_;
}
inline float FrameAnnotationTrackerCalculatorOptions::img_width() const {
  // @@protoc_insertion_point(field_get:mediapipe.FrameAnnotationTrackerCalculatorOptions.img_width)
  return _internal_img_width();
}
inline void FrameAnnotationTrackerCalculatorOptions::_internal_set_img_width(float value) {
  _has_bits_[0] |= 0x00000001u;
  img_width_ = value;
}
inline void FrameAnnotationTrackerCalculatorOptions::set_img_width(float value) {
  _internal_set_img_width(value);
  // @@protoc_insertion_point(field_set:mediapipe.FrameAnnotationTrackerCalculatorOptions.img_width)
}

// optional float img_height = 3;
inline bool FrameAnnotationTrackerCalculatorOptions::_internal_has_img_height() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool FrameAnnotationTrackerCalculatorOptions::has_img_height() const {
  return _internal_has_img_height();
}
inline void FrameAnnotationTrackerCalculatorOptions::clear_img_height() {
  img_height_ = 0;
  _has_bits_[0] &= ~0x00000002u;
}
inline float FrameAnnotationTrackerCalculatorOptions::_internal_img_height() const {
  return img_height_;
}
inline float FrameAnnotationTrackerCalculatorOptions::img_height() const {
  // @@protoc_insertion_point(field_get:mediapipe.FrameAnnotationTrackerCalculatorOptions.img_height)
  return _internal_img_height();
}
inline void FrameAnnotationTrackerCalculatorOptions::_internal_set_img_height(float value) {
  _has_bits_[0] |= 0x00000002u;
  img_height_ = value;
}
inline void FrameAnnotationTrackerCalculatorOptions::set_img_height(float value) {
  _internal_set_img_height(value);
  // @@protoc_insertion_point(field_set:mediapipe.FrameAnnotationTrackerCalculatorOptions.img_height)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fmodules_2fobjectron_2fcalculators_2fframe_5fannotation_5ftracker_5fcalculator_2eproto
