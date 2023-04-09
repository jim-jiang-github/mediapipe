// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/graphs/object_detection_3d/calculators/annotations_to_render_data_calculator.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fgraphs_2fobject_5fdetection_5f3d_2fcalculators_2fannotations_5fto_5frender_5fdata_5fcalculator_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fgraphs_2fobject_5fdetection_5f3d_2fcalculators_2fannotations_5fto_5frender_5fdata_5fcalculator_2eproto

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
#include "mediapipe/util/color.pb.h"
// @@protoc_insertion_point(includes)
#include <x/google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fgraphs_2fobject_5fdetection_5f3d_2fcalculators_2fannotations_5fto_5frender_5fdata_5fcalculator_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fgraphs_2fobject_5fdetection_5f3d_2fcalculators_2fannotations_5fto_5frender_5fdata_5fcalculator_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fgraphs_2fobject_5fdetection_5f3d_2fcalculators_2fannotations_5fto_5frender_5fdata_5fcalculator_2eproto;
namespace mediapipe {
class AnnotationsToRenderDataCalculatorOptions;
struct AnnotationsToRenderDataCalculatorOptionsDefaultTypeInternal;
extern AnnotationsToRenderDataCalculatorOptionsDefaultTypeInternal _AnnotationsToRenderDataCalculatorOptions_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::AnnotationsToRenderDataCalculatorOptions* Arena::CreateMaybeMessage<::mediapipe::AnnotationsToRenderDataCalculatorOptions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class AnnotationsToRenderDataCalculatorOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.AnnotationsToRenderDataCalculatorOptions) */ {
 public:
  inline AnnotationsToRenderDataCalculatorOptions() : AnnotationsToRenderDataCalculatorOptions(nullptr) {}
  ~AnnotationsToRenderDataCalculatorOptions() override;
  explicit constexpr AnnotationsToRenderDataCalculatorOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  AnnotationsToRenderDataCalculatorOptions(const AnnotationsToRenderDataCalculatorOptions& from);
  AnnotationsToRenderDataCalculatorOptions(AnnotationsToRenderDataCalculatorOptions&& from) noexcept
    : AnnotationsToRenderDataCalculatorOptions() {
    *this = ::std::move(from);
  }

  inline AnnotationsToRenderDataCalculatorOptions& operator=(const AnnotationsToRenderDataCalculatorOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline AnnotationsToRenderDataCalculatorOptions& operator=(AnnotationsToRenderDataCalculatorOptions&& from) noexcept {
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
  static const AnnotationsToRenderDataCalculatorOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const AnnotationsToRenderDataCalculatorOptions* internal_default_instance() {
    return reinterpret_cast<const AnnotationsToRenderDataCalculatorOptions*>(
               &_AnnotationsToRenderDataCalculatorOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(AnnotationsToRenderDataCalculatorOptions& a, AnnotationsToRenderDataCalculatorOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(AnnotationsToRenderDataCalculatorOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(AnnotationsToRenderDataCalculatorOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline AnnotationsToRenderDataCalculatorOptions* New() const final {
    return CreateMaybeMessage<AnnotationsToRenderDataCalculatorOptions>(nullptr);
  }

  AnnotationsToRenderDataCalculatorOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<AnnotationsToRenderDataCalculatorOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const AnnotationsToRenderDataCalculatorOptions& from);
  void MergeFrom(const AnnotationsToRenderDataCalculatorOptions& from);
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
  void InternalSwap(AnnotationsToRenderDataCalculatorOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.AnnotationsToRenderDataCalculatorOptions";
  }
  protected:
  explicit AnnotationsToRenderDataCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kLandmarkConnectionsFieldNumber = 1,
    kLandmarkColorFieldNumber = 2,
    kConnectionColorFieldNumber = 3,
    kVisualizeLandmarkDepthFieldNumber = 5,
    kThicknessFieldNumber = 4,
  };
  // repeated int32 landmark_connections = 1;
  int landmark_connections_size() const;
  private:
  int _internal_landmark_connections_size() const;
  public:
  void clear_landmark_connections();
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_landmark_connections(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >&
      _internal_landmark_connections() const;
  void _internal_add_landmark_connections(::PROTOBUF_NAMESPACE_ID::int32 value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >*
      _internal_mutable_landmark_connections();
  public:
  ::PROTOBUF_NAMESPACE_ID::int32 landmark_connections(int index) const;
  void set_landmark_connections(int index, ::PROTOBUF_NAMESPACE_ID::int32 value);
  void add_landmark_connections(::PROTOBUF_NAMESPACE_ID::int32 value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >&
      landmark_connections() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >*
      mutable_landmark_connections();

  // optional .mediapipe.Color landmark_color = 2;
  bool has_landmark_color() const;
  private:
  bool _internal_has_landmark_color() const;
  public:
  void clear_landmark_color();
  const ::mediapipe::Color& landmark_color() const;
  PROTOBUF_FUTURE_MUST_USE_RESULT ::mediapipe::Color* release_landmark_color();
  ::mediapipe::Color* mutable_landmark_color();
  void set_allocated_landmark_color(::mediapipe::Color* landmark_color);
  private:
  const ::mediapipe::Color& _internal_landmark_color() const;
  ::mediapipe::Color* _internal_mutable_landmark_color();
  public:
  void unsafe_arena_set_allocated_landmark_color(
      ::mediapipe::Color* landmark_color);
  ::mediapipe::Color* unsafe_arena_release_landmark_color();

  // optional .mediapipe.Color connection_color = 3;
  bool has_connection_color() const;
  private:
  bool _internal_has_connection_color() const;
  public:
  void clear_connection_color();
  const ::mediapipe::Color& connection_color() const;
  PROTOBUF_FUTURE_MUST_USE_RESULT ::mediapipe::Color* release_connection_color();
  ::mediapipe::Color* mutable_connection_color();
  void set_allocated_connection_color(::mediapipe::Color* connection_color);
  private:
  const ::mediapipe::Color& _internal_connection_color() const;
  ::mediapipe::Color* _internal_mutable_connection_color();
  public:
  void unsafe_arena_set_allocated_connection_color(
      ::mediapipe::Color* connection_color);
  ::mediapipe::Color* unsafe_arena_release_connection_color();

  // optional bool visualize_landmark_depth = 5 [default = true];
  bool has_visualize_landmark_depth() const;
  private:
  bool _internal_has_visualize_landmark_depth() const;
  public:
  void clear_visualize_landmark_depth();
  bool visualize_landmark_depth() const;
  void set_visualize_landmark_depth(bool value);
  private:
  bool _internal_visualize_landmark_depth() const;
  void _internal_set_visualize_landmark_depth(bool value);
  public:

  // optional double thickness = 4 [default = 1];
  bool has_thickness() const;
  private:
  bool _internal_has_thickness() const;
  public:
  void clear_thickness();
  double thickness() const;
  void set_thickness(double value);
  private:
  double _internal_thickness() const;
  void _internal_set_thickness(double value);
  public:

  static const int kExtFieldNumber = 267644238;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::AnnotationsToRenderDataCalculatorOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.AnnotationsToRenderDataCalculatorOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 > landmark_connections_;
  ::mediapipe::Color* landmark_color_;
  ::mediapipe::Color* connection_color_;
  bool visualize_landmark_depth_;
  double thickness_;
  friend struct ::TableStruct_mediapipe_2fgraphs_2fobject_5fdetection_5f3d_2fcalculators_2fannotations_5fto_5frender_5fdata_5fcalculator_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// AnnotationsToRenderDataCalculatorOptions

// repeated int32 landmark_connections = 1;
inline int AnnotationsToRenderDataCalculatorOptions::_internal_landmark_connections_size() const {
  return landmark_connections_.size();
}
inline int AnnotationsToRenderDataCalculatorOptions::landmark_connections_size() const {
  return _internal_landmark_connections_size();
}
inline void AnnotationsToRenderDataCalculatorOptions::clear_landmark_connections() {
  landmark_connections_.Clear();
}
inline ::PROTOBUF_NAMESPACE_ID::int32 AnnotationsToRenderDataCalculatorOptions::_internal_landmark_connections(int index) const {
  return landmark_connections_.Get(index);
}
inline ::PROTOBUF_NAMESPACE_ID::int32 AnnotationsToRenderDataCalculatorOptions::landmark_connections(int index) const {
  // @@protoc_insertion_point(field_get:mediapipe.AnnotationsToRenderDataCalculatorOptions.landmark_connections)
  return _internal_landmark_connections(index);
}
inline void AnnotationsToRenderDataCalculatorOptions::set_landmark_connections(int index, ::PROTOBUF_NAMESPACE_ID::int32 value) {
  landmark_connections_.Set(index, value);
  // @@protoc_insertion_point(field_set:mediapipe.AnnotationsToRenderDataCalculatorOptions.landmark_connections)
}
inline void AnnotationsToRenderDataCalculatorOptions::_internal_add_landmark_connections(::PROTOBUF_NAMESPACE_ID::int32 value) {
  landmark_connections_.Add(value);
}
inline void AnnotationsToRenderDataCalculatorOptions::add_landmark_connections(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_add_landmark_connections(value);
  // @@protoc_insertion_point(field_add:mediapipe.AnnotationsToRenderDataCalculatorOptions.landmark_connections)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >&
AnnotationsToRenderDataCalculatorOptions::_internal_landmark_connections() const {
  return landmark_connections_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >&
AnnotationsToRenderDataCalculatorOptions::landmark_connections() const {
  // @@protoc_insertion_point(field_list:mediapipe.AnnotationsToRenderDataCalculatorOptions.landmark_connections)
  return _internal_landmark_connections();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >*
AnnotationsToRenderDataCalculatorOptions::_internal_mutable_landmark_connections() {
  return &landmark_connections_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int32 >*
AnnotationsToRenderDataCalculatorOptions::mutable_landmark_connections() {
  // @@protoc_insertion_point(field_mutable_list:mediapipe.AnnotationsToRenderDataCalculatorOptions.landmark_connections)
  return _internal_mutable_landmark_connections();
}

// optional .mediapipe.Color landmark_color = 2;
inline bool AnnotationsToRenderDataCalculatorOptions::_internal_has_landmark_color() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  PROTOBUF_ASSUME(!value || landmark_color_ != nullptr);
  return value;
}
inline bool AnnotationsToRenderDataCalculatorOptions::has_landmark_color() const {
  return _internal_has_landmark_color();
}
inline const ::mediapipe::Color& AnnotationsToRenderDataCalculatorOptions::_internal_landmark_color() const {
  const ::mediapipe::Color* p = landmark_color_;
  return p != nullptr ? *p : reinterpret_cast<const ::mediapipe::Color&>(
      ::mediapipe::_Color_default_instance_);
}
inline const ::mediapipe::Color& AnnotationsToRenderDataCalculatorOptions::landmark_color() const {
  // @@protoc_insertion_point(field_get:mediapipe.AnnotationsToRenderDataCalculatorOptions.landmark_color)
  return _internal_landmark_color();
}
inline void AnnotationsToRenderDataCalculatorOptions::unsafe_arena_set_allocated_landmark_color(
    ::mediapipe::Color* landmark_color) {
  if (GetArena() == nullptr) {
    delete reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(landmark_color_);
  }
  landmark_color_ = landmark_color;
  if (landmark_color) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:mediapipe.AnnotationsToRenderDataCalculatorOptions.landmark_color)
}
inline ::mediapipe::Color* AnnotationsToRenderDataCalculatorOptions::release_landmark_color() {
  _has_bits_[0] &= ~0x00000001u;
  ::mediapipe::Color* temp = landmark_color_;
  landmark_color_ = nullptr;
  if (GetArena() != nullptr) {
    temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  }
  return temp;
}
inline ::mediapipe::Color* AnnotationsToRenderDataCalculatorOptions::unsafe_arena_release_landmark_color() {
  // @@protoc_insertion_point(field_release:mediapipe.AnnotationsToRenderDataCalculatorOptions.landmark_color)
  _has_bits_[0] &= ~0x00000001u;
  ::mediapipe::Color* temp = landmark_color_;
  landmark_color_ = nullptr;
  return temp;
}
inline ::mediapipe::Color* AnnotationsToRenderDataCalculatorOptions::_internal_mutable_landmark_color() {
  _has_bits_[0] |= 0x00000001u;
  if (landmark_color_ == nullptr) {
    auto* p = CreateMaybeMessage<::mediapipe::Color>(GetArena());
    landmark_color_ = p;
  }
  return landmark_color_;
}
inline ::mediapipe::Color* AnnotationsToRenderDataCalculatorOptions::mutable_landmark_color() {
  // @@protoc_insertion_point(field_mutable:mediapipe.AnnotationsToRenderDataCalculatorOptions.landmark_color)
  return _internal_mutable_landmark_color();
}
inline void AnnotationsToRenderDataCalculatorOptions::set_allocated_landmark_color(::mediapipe::Color* landmark_color) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArena();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(landmark_color_);
  }
  if (landmark_color) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
      reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(landmark_color)->GetArena();
    if (message_arena != submessage_arena) {
      landmark_color = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, landmark_color, submessage_arena);
    }
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  landmark_color_ = landmark_color;
  // @@protoc_insertion_point(field_set_allocated:mediapipe.AnnotationsToRenderDataCalculatorOptions.landmark_color)
}

// optional .mediapipe.Color connection_color = 3;
inline bool AnnotationsToRenderDataCalculatorOptions::_internal_has_connection_color() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  PROTOBUF_ASSUME(!value || connection_color_ != nullptr);
  return value;
}
inline bool AnnotationsToRenderDataCalculatorOptions::has_connection_color() const {
  return _internal_has_connection_color();
}
inline const ::mediapipe::Color& AnnotationsToRenderDataCalculatorOptions::_internal_connection_color() const {
  const ::mediapipe::Color* p = connection_color_;
  return p != nullptr ? *p : reinterpret_cast<const ::mediapipe::Color&>(
      ::mediapipe::_Color_default_instance_);
}
inline const ::mediapipe::Color& AnnotationsToRenderDataCalculatorOptions::connection_color() const {
  // @@protoc_insertion_point(field_get:mediapipe.AnnotationsToRenderDataCalculatorOptions.connection_color)
  return _internal_connection_color();
}
inline void AnnotationsToRenderDataCalculatorOptions::unsafe_arena_set_allocated_connection_color(
    ::mediapipe::Color* connection_color) {
  if (GetArena() == nullptr) {
    delete reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(connection_color_);
  }
  connection_color_ = connection_color;
  if (connection_color) {
    _has_bits_[0] |= 0x00000002u;
  } else {
    _has_bits_[0] &= ~0x00000002u;
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:mediapipe.AnnotationsToRenderDataCalculatorOptions.connection_color)
}
inline ::mediapipe::Color* AnnotationsToRenderDataCalculatorOptions::release_connection_color() {
  _has_bits_[0] &= ~0x00000002u;
  ::mediapipe::Color* temp = connection_color_;
  connection_color_ = nullptr;
  if (GetArena() != nullptr) {
    temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  }
  return temp;
}
inline ::mediapipe::Color* AnnotationsToRenderDataCalculatorOptions::unsafe_arena_release_connection_color() {
  // @@protoc_insertion_point(field_release:mediapipe.AnnotationsToRenderDataCalculatorOptions.connection_color)
  _has_bits_[0] &= ~0x00000002u;
  ::mediapipe::Color* temp = connection_color_;
  connection_color_ = nullptr;
  return temp;
}
inline ::mediapipe::Color* AnnotationsToRenderDataCalculatorOptions::_internal_mutable_connection_color() {
  _has_bits_[0] |= 0x00000002u;
  if (connection_color_ == nullptr) {
    auto* p = CreateMaybeMessage<::mediapipe::Color>(GetArena());
    connection_color_ = p;
  }
  return connection_color_;
}
inline ::mediapipe::Color* AnnotationsToRenderDataCalculatorOptions::mutable_connection_color() {
  // @@protoc_insertion_point(field_mutable:mediapipe.AnnotationsToRenderDataCalculatorOptions.connection_color)
  return _internal_mutable_connection_color();
}
inline void AnnotationsToRenderDataCalculatorOptions::set_allocated_connection_color(::mediapipe::Color* connection_color) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArena();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(connection_color_);
  }
  if (connection_color) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
      reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(connection_color)->GetArena();
    if (message_arena != submessage_arena) {
      connection_color = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, connection_color, submessage_arena);
    }
    _has_bits_[0] |= 0x00000002u;
  } else {
    _has_bits_[0] &= ~0x00000002u;
  }
  connection_color_ = connection_color;
  // @@protoc_insertion_point(field_set_allocated:mediapipe.AnnotationsToRenderDataCalculatorOptions.connection_color)
}

// optional double thickness = 4 [default = 1];
inline bool AnnotationsToRenderDataCalculatorOptions::_internal_has_thickness() const {
  bool value = (_has_bits_[0] & 0x00000008u) != 0;
  return value;
}
inline bool AnnotationsToRenderDataCalculatorOptions::has_thickness() const {
  return _internal_has_thickness();
}
inline void AnnotationsToRenderDataCalculatorOptions::clear_thickness() {
  thickness_ = 1;
  _has_bits_[0] &= ~0x00000008u;
}
inline double AnnotationsToRenderDataCalculatorOptions::_internal_thickness() const {
  return thickness_;
}
inline double AnnotationsToRenderDataCalculatorOptions::thickness() const {
  // @@protoc_insertion_point(field_get:mediapipe.AnnotationsToRenderDataCalculatorOptions.thickness)
  return _internal_thickness();
}
inline void AnnotationsToRenderDataCalculatorOptions::_internal_set_thickness(double value) {
  _has_bits_[0] |= 0x00000008u;
  thickness_ = value;
}
inline void AnnotationsToRenderDataCalculatorOptions::set_thickness(double value) {
  _internal_set_thickness(value);
  // @@protoc_insertion_point(field_set:mediapipe.AnnotationsToRenderDataCalculatorOptions.thickness)
}

// optional bool visualize_landmark_depth = 5 [default = true];
inline bool AnnotationsToRenderDataCalculatorOptions::_internal_has_visualize_landmark_depth() const {
  bool value = (_has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool AnnotationsToRenderDataCalculatorOptions::has_visualize_landmark_depth() const {
  return _internal_has_visualize_landmark_depth();
}
inline void AnnotationsToRenderDataCalculatorOptions::clear_visualize_landmark_depth() {
  visualize_landmark_depth_ = true;
  _has_bits_[0] &= ~0x00000004u;
}
inline bool AnnotationsToRenderDataCalculatorOptions::_internal_visualize_landmark_depth() const {
  return visualize_landmark_depth_;
}
inline bool AnnotationsToRenderDataCalculatorOptions::visualize_landmark_depth() const {
  // @@protoc_insertion_point(field_get:mediapipe.AnnotationsToRenderDataCalculatorOptions.visualize_landmark_depth)
  return _internal_visualize_landmark_depth();
}
inline void AnnotationsToRenderDataCalculatorOptions::_internal_set_visualize_landmark_depth(bool value) {
  _has_bits_[0] |= 0x00000004u;
  visualize_landmark_depth_ = value;
}
inline void AnnotationsToRenderDataCalculatorOptions::set_visualize_landmark_depth(bool value) {
  _internal_set_visualize_landmark_depth(value);
  // @@protoc_insertion_point(field_set:mediapipe.AnnotationsToRenderDataCalculatorOptions.visualize_landmark_depth)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <x/google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fgraphs_2fobject_5fdetection_5f3d_2fcalculators_2fannotations_5fto_5frender_5fdata_5fcalculator_2eproto
