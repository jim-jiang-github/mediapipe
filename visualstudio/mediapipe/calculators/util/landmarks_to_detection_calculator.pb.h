// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/util/landmarks_to_detection_calculator.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5fdetection_5fcalculator_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5fdetection_5fcalculator_2eproto

#include <limits>
#include <string>

#include <x/google/protobuf/port_def.inc>
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

#include <x/google/protobuf/port_undef.inc>
#include <x/google/protobuf/io/coded_stream.h>
#include <x/google/protobuf/arena.h>
#include <x/google/protobuf/arenastring.h>
#include <x/google/protobuf/generated_message_table_driven.h>
#include <x/google/protobuf/generated_message_util.h>
#include <x/google/protobuf/metadata_lite.h>
#include <x/google/protobuf/generated_message_reflection.h>
#include <x/google/protobuf/message.h>
#include <x/google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <x/google/protobuf/extension_set.h>  // IWYU pragma: export
#include <x/google/protobuf/unknown_field_set.h>
#include "mediapipe/framework/calculator.pb.h"
// @@protoc_insertion_point(includes)
#include <x/google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5fdetection_5fcalculator_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5fdetection_5fcalculator_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5fdetection_5fcalculator_2eproto;
namespace mediapipe {
class LandmarksToDetectionCalculatorOptions;
struct LandmarksToDetectionCalculatorOptionsDefaultTypeInternal;
extern LandmarksToDetectionCalculatorOptionsDefaultTypeInternal _LandmarksToDetectionCalculatorOptions_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::LandmarksToDetectionCalculatorOptions* Arena::CreateMaybeMessage<::mediapipe::LandmarksToDetectionCalculatorOptions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class LandmarksToDetectionCalculatorOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.LandmarksToDetectionCalculatorOptions) */ {
 public:
  inline LandmarksToDetectionCalculatorOptions() : LandmarksToDetectionCalculatorOptions(nullptr) {}
  ~LandmarksToDetectionCalculatorOptions() override;
  explicit constexpr LandmarksToDetectionCalculatorOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  LandmarksToDetectionCalculatorOptions(const LandmarksToDetectionCalculatorOptions& from);
  LandmarksToDetectionCalculatorOptions(LandmarksToDetectionCalculatorOptions&& from) noexcept
    : LandmarksToDetectionCalculatorOptions() {
    *this = ::std::move(from);
  }

  inline LandmarksToDetectionCalculatorOptions& operator=(const LandmarksToDetectionCalculatorOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline LandmarksToDetectionCalculatorOptions& operator=(LandmarksToDetectionCalculatorOptions&& from) noexcept {
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
  static const LandmarksToDetectionCalculatorOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const LandmarksToDetectionCalculatorOptions* internal_default_instance() {
    return reinterpret_cast<const LandmarksToDetectionCalculatorOptions*>(
               &_LandmarksToDetectionCalculatorOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(LandmarksToDetectionCalculatorOptions& a, LandmarksToDetectionCalculatorOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(LandmarksToDetectionCalculatorOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(LandmarksToDetectionCalculatorOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline LandmarksToDetectionCalculatorOptions* New() const final {
    return CreateMaybeMessage<LandmarksToDetectionCalculatorOptions>(nullptr);
  }

  LandmarksToDetectionCalculatorOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<LandmarksToDetectionCalculatorOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const LandmarksToDetectionCalculatorOptions& from);
  void MergeFrom(const LandmarksToDetectionCalculatorOptions& from);
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
  void InternalSwap(LandmarksToDetectionCalculatorOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.LandmarksToDetectionCalculatorOptions";
  }
  protected:
  explicit LandmarksToDetectionCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kSelectedLandmarkIndicesFieldNumber = 1,
  };
  // repeated int32 selected_landmark_indices = 1;
  int selected_landmark_indices_size() const;
  private:
  int _internal_selected_landmark_indices_size() const;
  public:
  void clear_selected_landmark_indices();
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_selected_landmark_indices(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >&
      _internal_selected_landmark_indices() const;
  void _internal_add_selected_landmark_indices(::PROTOBUF_NAMESPACE_ID::int32 value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >*
      _internal_mutable_selected_landmark_indices();
  public:
  ::PROTOBUF_NAMESPACE_ID::int32 selected_landmark_indices(int index) const;
  void set_selected_landmark_indices(int index, ::PROTOBUF_NAMESPACE_ID::int32 value);
  void add_selected_landmark_indices(::PROTOBUF_NAMESPACE_ID::int32 value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >&
      selected_landmark_indices() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >*
      mutable_selected_landmark_indices();

  static const int kExtFieldNumber = 260199669;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::LandmarksToDetectionCalculatorOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.LandmarksToDetectionCalculatorOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 > selected_landmark_indices_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5fdetection_5fcalculator_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// LandmarksToDetectionCalculatorOptions

// repeated int32 selected_landmark_indices = 1;
inline int LandmarksToDetectionCalculatorOptions::_internal_selected_landmark_indices_size() const {
  return selected_landmark_indices_.size();
}
inline int LandmarksToDetectionCalculatorOptions::selected_landmark_indices_size() const {
  return _internal_selected_landmark_indices_size();
}
inline void LandmarksToDetectionCalculatorOptions::clear_selected_landmark_indices() {
  selected_landmark_indices_.Clear();
}
inline ::PROTOBUF_NAMESPACE_ID::int32 LandmarksToDetectionCalculatorOptions::_internal_selected_landmark_indices(int index) const {
  return selected_landmark_indices_.Get(index);
}
inline ::PROTOBUF_NAMESPACE_ID::int32 LandmarksToDetectionCalculatorOptions::selected_landmark_indices(int index) const {
  // @@protoc_insertion_point(field_get:mediapipe.LandmarksToDetectionCalculatorOptions.selected_landmark_indices)
  return _internal_selected_landmark_indices(index);
}
inline void LandmarksToDetectionCalculatorOptions::set_selected_landmark_indices(int index, ::PROTOBUF_NAMESPACE_ID::int32 value) {
  selected_landmark_indices_.Set(index, value);
  // @@protoc_insertion_point(field_set:mediapipe.LandmarksToDetectionCalculatorOptions.selected_landmark_indices)
}
inline void LandmarksToDetectionCalculatorOptions::_internal_add_selected_landmark_indices(::PROTOBUF_NAMESPACE_ID::int32 value) {
  selected_landmark_indices_.Add(value);
}
inline void LandmarksToDetectionCalculatorOptions::add_selected_landmark_indices(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_add_selected_landmark_indices(value);
  // @@protoc_insertion_point(field_add:mediapipe.LandmarksToDetectionCalculatorOptions.selected_landmark_indices)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >&
LandmarksToDetectionCalculatorOptions::_internal_selected_landmark_indices() const {
  return selected_landmark_indices_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >&
LandmarksToDetectionCalculatorOptions::selected_landmark_indices() const {
  // @@protoc_insertion_point(field_list:mediapipe.LandmarksToDetectionCalculatorOptions.selected_landmark_indices)
  return _internal_selected_landmark_indices();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >*
LandmarksToDetectionCalculatorOptions::_internal_mutable_selected_landmark_indices() {
  return &selected_landmark_indices_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >*
LandmarksToDetectionCalculatorOptions::mutable_selected_landmark_indices() {
  // @@protoc_insertion_point(field_mutable_list:mediapipe.LandmarksToDetectionCalculatorOptions.selected_landmark_indices)
  return _internal_mutable_selected_landmark_indices();
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <x/google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5fdetection_5fcalculator_2eproto
