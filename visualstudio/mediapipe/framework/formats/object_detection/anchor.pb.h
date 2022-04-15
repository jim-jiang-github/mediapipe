// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/framework/formats/object_detection/anchor.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fframework_2fformats_2fobject_5fdetection_2fanchor_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fframework_2fformats_2fobject_5fdetection_2fanchor_2eproto

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
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fframework_2fformats_2fobject_5fdetection_2fanchor_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fframework_2fformats_2fobject_5fdetection_2fanchor_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fframework_2fformats_2fobject_5fdetection_2fanchor_2eproto;
namespace mediapipe {
class Anchor;
struct AnchorDefaultTypeInternal;
extern AnchorDefaultTypeInternal _Anchor_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::Anchor* Arena::CreateMaybeMessage<::mediapipe::Anchor>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class Anchor PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.Anchor) */ {
 public:
  inline Anchor() : Anchor(nullptr) {}
  ~Anchor() override;
  explicit constexpr Anchor(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  Anchor(const Anchor& from);
  Anchor(Anchor&& from) noexcept
    : Anchor() {
    *this = ::std::move(from);
  }

  inline Anchor& operator=(const Anchor& from) {
    CopyFrom(from);
    return *this;
  }
  inline Anchor& operator=(Anchor&& from) noexcept {
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
  static const Anchor& default_instance() {
    return *internal_default_instance();
  }
  static inline const Anchor* internal_default_instance() {
    return reinterpret_cast<const Anchor*>(
               &_Anchor_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(Anchor& a, Anchor& b) {
    a.Swap(&b);
  }
  inline void Swap(Anchor* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(Anchor* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline Anchor* New() const final {
    return CreateMaybeMessage<Anchor>(nullptr);
  }

  Anchor* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<Anchor>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const Anchor& from);
  void MergeFrom(const Anchor& from);
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
  void InternalSwap(Anchor* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.Anchor";
  }
  protected:
  explicit Anchor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kXCenterFieldNumber = 1,
    kYCenterFieldNumber = 2,
    kHFieldNumber = 3,
    kWFieldNumber = 4,
  };
  // required float x_center = 1;
  bool has_x_center() const;
  private:
  bool _internal_has_x_center() const;
  public:
  void clear_x_center();
  float x_center() const;
  void set_x_center(float value);
  private:
  float _internal_x_center() const;
  void _internal_set_x_center(float value);
  public:

  // required float y_center = 2;
  bool has_y_center() const;
  private:
  bool _internal_has_y_center() const;
  public:
  void clear_y_center();
  float y_center() const;
  void set_y_center(float value);
  private:
  float _internal_y_center() const;
  void _internal_set_y_center(float value);
  public:

  // required float h = 3;
  bool has_h() const;
  private:
  bool _internal_has_h() const;
  public:
  void clear_h();
  float h() const;
  void set_h(float value);
  private:
  float _internal_h() const;
  void _internal_set_h(float value);
  public:

  // required float w = 4;
  bool has_w() const;
  private:
  bool _internal_has_w() const;
  public:
  void clear_w();
  float w() const;
  void set_w(float value);
  private:
  float _internal_w() const;
  void _internal_set_w(float value);
  public:

  // @@protoc_insertion_point(class_scope:mediapipe.Anchor)
 private:
  class _Internal;

  // helper for ByteSizeLong()
  size_t RequiredFieldsByteSizeFallback() const;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  float x_center_;
  float y_center_;
  float h_;
  float w_;
  friend struct ::TableStruct_mediapipe_2fframework_2fformats_2fobject_5fdetection_2fanchor_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// Anchor

// required float x_center = 1;
inline bool Anchor::_internal_has_x_center() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool Anchor::has_x_center() const {
  return _internal_has_x_center();
}
inline void Anchor::clear_x_center() {
  x_center_ = 0;
  _has_bits_[0] &= ~0x00000001u;
}
inline float Anchor::_internal_x_center() const {
  return x_center_;
}
inline float Anchor::x_center() const {
  // @@protoc_insertion_point(field_get:mediapipe.Anchor.x_center)
  return _internal_x_center();
}
inline void Anchor::_internal_set_x_center(float value) {
  _has_bits_[0] |= 0x00000001u;
  x_center_ = value;
}
inline void Anchor::set_x_center(float value) {
  _internal_set_x_center(value);
  // @@protoc_insertion_point(field_set:mediapipe.Anchor.x_center)
}

// required float y_center = 2;
inline bool Anchor::_internal_has_y_center() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool Anchor::has_y_center() const {
  return _internal_has_y_center();
}
inline void Anchor::clear_y_center() {
  y_center_ = 0;
  _has_bits_[0] &= ~0x00000002u;
}
inline float Anchor::_internal_y_center() const {
  return y_center_;
}
inline float Anchor::y_center() const {
  // @@protoc_insertion_point(field_get:mediapipe.Anchor.y_center)
  return _internal_y_center();
}
inline void Anchor::_internal_set_y_center(float value) {
  _has_bits_[0] |= 0x00000002u;
  y_center_ = value;
}
inline void Anchor::set_y_center(float value) {
  _internal_set_y_center(value);
  // @@protoc_insertion_point(field_set:mediapipe.Anchor.y_center)
}

// required float h = 3;
inline bool Anchor::_internal_has_h() const {
  bool value = (_has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool Anchor::has_h() const {
  return _internal_has_h();
}
inline void Anchor::clear_h() {
  h_ = 0;
  _has_bits_[0] &= ~0x00000004u;
}
inline float Anchor::_internal_h() const {
  return h_;
}
inline float Anchor::h() const {
  // @@protoc_insertion_point(field_get:mediapipe.Anchor.h)
  return _internal_h();
}
inline void Anchor::_internal_set_h(float value) {
  _has_bits_[0] |= 0x00000004u;
  h_ = value;
}
inline void Anchor::set_h(float value) {
  _internal_set_h(value);
  // @@protoc_insertion_point(field_set:mediapipe.Anchor.h)
}

// required float w = 4;
inline bool Anchor::_internal_has_w() const {
  bool value = (_has_bits_[0] & 0x00000008u) != 0;
  return value;
}
inline bool Anchor::has_w() const {
  return _internal_has_w();
}
inline void Anchor::clear_w() {
  w_ = 0;
  _has_bits_[0] &= ~0x00000008u;
}
inline float Anchor::_internal_w() const {
  return w_;
}
inline float Anchor::w() const {
  // @@protoc_insertion_point(field_get:mediapipe.Anchor.w)
  return _internal_w();
}
inline void Anchor::_internal_set_w(float value) {
  _has_bits_[0] |= 0x00000008u;
  w_ = value;
}
inline void Anchor::set_w(float value) {
  _internal_set_w(value);
  // @@protoc_insertion_point(field_set:mediapipe.Anchor.w)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fframework_2fformats_2fobject_5fdetection_2fanchor_2eproto
