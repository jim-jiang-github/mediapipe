// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/gpu/gl_animation_overlay_calculator.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fgpu_2fgl_5fanimation_5foverlay_5fcalculator_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fgpu_2fgl_5fanimation_5foverlay_5fcalculator_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fgpu_2fgl_5fanimation_5foverlay_5fcalculator_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fgpu_2fgl_5fanimation_5foverlay_5fcalculator_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fgpu_2fgl_5fanimation_5foverlay_5fcalculator_2eproto;
namespace mediapipe {
class GlAnimationOverlayCalculatorOptions;
struct GlAnimationOverlayCalculatorOptionsDefaultTypeInternal;
extern GlAnimationOverlayCalculatorOptionsDefaultTypeInternal _GlAnimationOverlayCalculatorOptions_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::GlAnimationOverlayCalculatorOptions* Arena::CreateMaybeMessage<::mediapipe::GlAnimationOverlayCalculatorOptions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class GlAnimationOverlayCalculatorOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.GlAnimationOverlayCalculatorOptions) */ {
 public:
  inline GlAnimationOverlayCalculatorOptions() : GlAnimationOverlayCalculatorOptions(nullptr) {}
  ~GlAnimationOverlayCalculatorOptions() override;
  explicit constexpr GlAnimationOverlayCalculatorOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  GlAnimationOverlayCalculatorOptions(const GlAnimationOverlayCalculatorOptions& from);
  GlAnimationOverlayCalculatorOptions(GlAnimationOverlayCalculatorOptions&& from) noexcept
    : GlAnimationOverlayCalculatorOptions() {
    *this = ::std::move(from);
  }

  inline GlAnimationOverlayCalculatorOptions& operator=(const GlAnimationOverlayCalculatorOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline GlAnimationOverlayCalculatorOptions& operator=(GlAnimationOverlayCalculatorOptions&& from) noexcept {
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
  static const GlAnimationOverlayCalculatorOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const GlAnimationOverlayCalculatorOptions* internal_default_instance() {
    return reinterpret_cast<const GlAnimationOverlayCalculatorOptions*>(
               &_GlAnimationOverlayCalculatorOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(GlAnimationOverlayCalculatorOptions& a, GlAnimationOverlayCalculatorOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(GlAnimationOverlayCalculatorOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(GlAnimationOverlayCalculatorOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline GlAnimationOverlayCalculatorOptions* New() const final {
    return CreateMaybeMessage<GlAnimationOverlayCalculatorOptions>(nullptr);
  }

  GlAnimationOverlayCalculatorOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<GlAnimationOverlayCalculatorOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const GlAnimationOverlayCalculatorOptions& from);
  void MergeFrom(const GlAnimationOverlayCalculatorOptions& from);
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
  void InternalSwap(GlAnimationOverlayCalculatorOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.GlAnimationOverlayCalculatorOptions";
  }
  protected:
  explicit GlAnimationOverlayCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kAnimationSpeedFpsFieldNumber = 5,
    kAspectRatioFieldNumber = 1,
    kVerticalFovDegreesFieldNumber = 2,
    kZClippingPlaneNearFieldNumber = 3,
    kZClippingPlaneFarFieldNumber = 4,
  };
  // optional float animation_speed_fps = 5 [default = 25];
  bool has_animation_speed_fps() const;
  private:
  bool _internal_has_animation_speed_fps() const;
  public:
  void clear_animation_speed_fps();
  float animation_speed_fps() const;
  void set_animation_speed_fps(float value);
  private:
  float _internal_animation_speed_fps() const;
  void _internal_set_animation_speed_fps(float value);
  public:

  // optional float aspect_ratio = 1 [default = 0.75];
  bool has_aspect_ratio() const;
  private:
  bool _internal_has_aspect_ratio() const;
  public:
  void clear_aspect_ratio();
  float aspect_ratio() const;
  void set_aspect_ratio(float value);
  private:
  float _internal_aspect_ratio() const;
  void _internal_set_aspect_ratio(float value);
  public:

  // optional float vertical_fov_degrees = 2 [default = 70];
  bool has_vertical_fov_degrees() const;
  private:
  bool _internal_has_vertical_fov_degrees() const;
  public:
  void clear_vertical_fov_degrees();
  float vertical_fov_degrees() const;
  void set_vertical_fov_degrees(float value);
  private:
  float _internal_vertical_fov_degrees() const;
  void _internal_set_vertical_fov_degrees(float value);
  public:

  // optional float z_clipping_plane_near = 3 [default = 0.1];
  bool has_z_clipping_plane_near() const;
  private:
  bool _internal_has_z_clipping_plane_near() const;
  public:
  void clear_z_clipping_plane_near();
  float z_clipping_plane_near() const;
  void set_z_clipping_plane_near(float value);
  private:
  float _internal_z_clipping_plane_near() const;
  void _internal_set_z_clipping_plane_near(float value);
  public:

  // optional float z_clipping_plane_far = 4 [default = 1000];
  bool has_z_clipping_plane_far() const;
  private:
  bool _internal_has_z_clipping_plane_far() const;
  public:
  void clear_z_clipping_plane_far();
  float z_clipping_plane_far() const;
  void set_z_clipping_plane_far(float value);
  private:
  float _internal_z_clipping_plane_far() const;
  void _internal_set_z_clipping_plane_far(float value);
  public:

  static const int kExtFieldNumber = 174760573;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::GlAnimationOverlayCalculatorOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.GlAnimationOverlayCalculatorOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  float animation_speed_fps_;
  float aspect_ratio_;
  float vertical_fov_degrees_;
  float z_clipping_plane_near_;
  float z_clipping_plane_far_;
  friend struct ::TableStruct_mediapipe_2fgpu_2fgl_5fanimation_5foverlay_5fcalculator_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// GlAnimationOverlayCalculatorOptions

// optional float aspect_ratio = 1 [default = 0.75];
inline bool GlAnimationOverlayCalculatorOptions::_internal_has_aspect_ratio() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool GlAnimationOverlayCalculatorOptions::has_aspect_ratio() const {
  return _internal_has_aspect_ratio();
}
inline void GlAnimationOverlayCalculatorOptions::clear_aspect_ratio() {
  aspect_ratio_ = 0.75f;
  _has_bits_[0] &= ~0x00000002u;
}
inline float GlAnimationOverlayCalculatorOptions::_internal_aspect_ratio() const {
  return aspect_ratio_;
}
inline float GlAnimationOverlayCalculatorOptions::aspect_ratio() const {
  // @@protoc_insertion_point(field_get:mediapipe.GlAnimationOverlayCalculatorOptions.aspect_ratio)
  return _internal_aspect_ratio();
}
inline void GlAnimationOverlayCalculatorOptions::_internal_set_aspect_ratio(float value) {
  _has_bits_[0] |= 0x00000002u;
  aspect_ratio_ = value;
}
inline void GlAnimationOverlayCalculatorOptions::set_aspect_ratio(float value) {
  _internal_set_aspect_ratio(value);
  // @@protoc_insertion_point(field_set:mediapipe.GlAnimationOverlayCalculatorOptions.aspect_ratio)
}

// optional float vertical_fov_degrees = 2 [default = 70];
inline bool GlAnimationOverlayCalculatorOptions::_internal_has_vertical_fov_degrees() const {
  bool value = (_has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool GlAnimationOverlayCalculatorOptions::has_vertical_fov_degrees() const {
  return _internal_has_vertical_fov_degrees();
}
inline void GlAnimationOverlayCalculatorOptions::clear_vertical_fov_degrees() {
  vertical_fov_degrees_ = 70;
  _has_bits_[0] &= ~0x00000004u;
}
inline float GlAnimationOverlayCalculatorOptions::_internal_vertical_fov_degrees() const {
  return vertical_fov_degrees_;
}
inline float GlAnimationOverlayCalculatorOptions::vertical_fov_degrees() const {
  // @@protoc_insertion_point(field_get:mediapipe.GlAnimationOverlayCalculatorOptions.vertical_fov_degrees)
  return _internal_vertical_fov_degrees();
}
inline void GlAnimationOverlayCalculatorOptions::_internal_set_vertical_fov_degrees(float value) {
  _has_bits_[0] |= 0x00000004u;
  vertical_fov_degrees_ = value;
}
inline void GlAnimationOverlayCalculatorOptions::set_vertical_fov_degrees(float value) {
  _internal_set_vertical_fov_degrees(value);
  // @@protoc_insertion_point(field_set:mediapipe.GlAnimationOverlayCalculatorOptions.vertical_fov_degrees)
}

// optional float z_clipping_plane_near = 3 [default = 0.1];
inline bool GlAnimationOverlayCalculatorOptions::_internal_has_z_clipping_plane_near() const {
  bool value = (_has_bits_[0] & 0x00000008u) != 0;
  return value;
}
inline bool GlAnimationOverlayCalculatorOptions::has_z_clipping_plane_near() const {
  return _internal_has_z_clipping_plane_near();
}
inline void GlAnimationOverlayCalculatorOptions::clear_z_clipping_plane_near() {
  z_clipping_plane_near_ = 0.1f;
  _has_bits_[0] &= ~0x00000008u;
}
inline float GlAnimationOverlayCalculatorOptions::_internal_z_clipping_plane_near() const {
  return z_clipping_plane_near_;
}
inline float GlAnimationOverlayCalculatorOptions::z_clipping_plane_near() const {
  // @@protoc_insertion_point(field_get:mediapipe.GlAnimationOverlayCalculatorOptions.z_clipping_plane_near)
  return _internal_z_clipping_plane_near();
}
inline void GlAnimationOverlayCalculatorOptions::_internal_set_z_clipping_plane_near(float value) {
  _has_bits_[0] |= 0x00000008u;
  z_clipping_plane_near_ = value;
}
inline void GlAnimationOverlayCalculatorOptions::set_z_clipping_plane_near(float value) {
  _internal_set_z_clipping_plane_near(value);
  // @@protoc_insertion_point(field_set:mediapipe.GlAnimationOverlayCalculatorOptions.z_clipping_plane_near)
}

// optional float z_clipping_plane_far = 4 [default = 1000];
inline bool GlAnimationOverlayCalculatorOptions::_internal_has_z_clipping_plane_far() const {
  bool value = (_has_bits_[0] & 0x00000010u) != 0;
  return value;
}
inline bool GlAnimationOverlayCalculatorOptions::has_z_clipping_plane_far() const {
  return _internal_has_z_clipping_plane_far();
}
inline void GlAnimationOverlayCalculatorOptions::clear_z_clipping_plane_far() {
  z_clipping_plane_far_ = 1000;
  _has_bits_[0] &= ~0x00000010u;
}
inline float GlAnimationOverlayCalculatorOptions::_internal_z_clipping_plane_far() const {
  return z_clipping_plane_far_;
}
inline float GlAnimationOverlayCalculatorOptions::z_clipping_plane_far() const {
  // @@protoc_insertion_point(field_get:mediapipe.GlAnimationOverlayCalculatorOptions.z_clipping_plane_far)
  return _internal_z_clipping_plane_far();
}
inline void GlAnimationOverlayCalculatorOptions::_internal_set_z_clipping_plane_far(float value) {
  _has_bits_[0] |= 0x00000010u;
  z_clipping_plane_far_ = value;
}
inline void GlAnimationOverlayCalculatorOptions::set_z_clipping_plane_far(float value) {
  _internal_set_z_clipping_plane_far(value);
  // @@protoc_insertion_point(field_set:mediapipe.GlAnimationOverlayCalculatorOptions.z_clipping_plane_far)
}

// optional float animation_speed_fps = 5 [default = 25];
inline bool GlAnimationOverlayCalculatorOptions::_internal_has_animation_speed_fps() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool GlAnimationOverlayCalculatorOptions::has_animation_speed_fps() const {
  return _internal_has_animation_speed_fps();
}
inline void GlAnimationOverlayCalculatorOptions::clear_animation_speed_fps() {
  animation_speed_fps_ = 25;
  _has_bits_[0] &= ~0x00000001u;
}
inline float GlAnimationOverlayCalculatorOptions::_internal_animation_speed_fps() const {
  return animation_speed_fps_;
}
inline float GlAnimationOverlayCalculatorOptions::animation_speed_fps() const {
  // @@protoc_insertion_point(field_get:mediapipe.GlAnimationOverlayCalculatorOptions.animation_speed_fps)
  return _internal_animation_speed_fps();
}
inline void GlAnimationOverlayCalculatorOptions::_internal_set_animation_speed_fps(float value) {
  _has_bits_[0] |= 0x00000001u;
  animation_speed_fps_ = value;
}
inline void GlAnimationOverlayCalculatorOptions::set_animation_speed_fps(float value) {
  _internal_set_animation_speed_fps(value);
  // @@protoc_insertion_point(field_set:mediapipe.GlAnimationOverlayCalculatorOptions.animation_speed_fps)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fgpu_2fgl_5fanimation_5foverlay_5fcalculator_2eproto
