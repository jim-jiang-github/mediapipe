// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/util/tracking/push_pull_filtering.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2futil_2ftracking_2fpush_5fpull_5ffiltering_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2futil_2ftracking_2fpush_5fpull_5ffiltering_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2futil_2ftracking_2fpush_5fpull_5ffiltering_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2futil_2ftracking_2fpush_5fpull_5ffiltering_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2futil_2ftracking_2fpush_5fpull_5ffiltering_2eproto;
namespace mediapipe {
class PushPullOptions;
struct PushPullOptionsDefaultTypeInternal;
extern PushPullOptionsDefaultTypeInternal _PushPullOptions_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::PushPullOptions* Arena::CreateMaybeMessage<::mediapipe::PushPullOptions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class PushPullOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.PushPullOptions) */ {
 public:
  inline PushPullOptions() : PushPullOptions(nullptr) {}
  ~PushPullOptions() override;
  explicit constexpr PushPullOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  PushPullOptions(const PushPullOptions& from);
  PushPullOptions(PushPullOptions&& from) noexcept
    : PushPullOptions() {
    *this = ::std::move(from);
  }

  inline PushPullOptions& operator=(const PushPullOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline PushPullOptions& operator=(PushPullOptions&& from) noexcept {
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
  static const PushPullOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const PushPullOptions* internal_default_instance() {
    return reinterpret_cast<const PushPullOptions*>(
               &_PushPullOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(PushPullOptions& a, PushPullOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(PushPullOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(PushPullOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline PushPullOptions* New() const final {
    return CreateMaybeMessage<PushPullOptions>(nullptr);
  }

  PushPullOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<PushPullOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const PushPullOptions& from);
  void MergeFrom(const PushPullOptions& from);
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
  void InternalSwap(PushPullOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.PushPullOptions";
  }
  protected:
  explicit PushPullOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kPushBilateralScaleFieldNumber = 6,
    kBilateralSigmaFieldNumber = 1,
    kPullPropagationScaleFieldNumber = 3,
    kPushPropagationScaleFieldNumber = 4,
    kPullBilateralScaleFieldNumber = 5,
  };
  // optional float push_bilateral_scale = 6 [default = 0.9];
  bool has_push_bilateral_scale() const;
  private:
  bool _internal_has_push_bilateral_scale() const;
  public:
  void clear_push_bilateral_scale();
  float push_bilateral_scale() const;
  void set_push_bilateral_scale(float value);
  private:
  float _internal_push_bilateral_scale() const;
  void _internal_set_push_bilateral_scale(float value);
  public:

  // optional float bilateral_sigma = 1 [default = 20];
  bool has_bilateral_sigma() const;
  private:
  bool _internal_has_bilateral_sigma() const;
  public:
  void clear_bilateral_sigma();
  float bilateral_sigma() const;
  void set_bilateral_sigma(float value);
  private:
  float _internal_bilateral_sigma() const;
  void _internal_set_bilateral_sigma(float value);
  public:

  // optional float pull_propagation_scale = 3 [default = 8];
  bool has_pull_propagation_scale() const;
  private:
  bool _internal_has_pull_propagation_scale() const;
  public:
  void clear_pull_propagation_scale();
  float pull_propagation_scale() const;
  void set_pull_propagation_scale(float value);
  private:
  float _internal_pull_propagation_scale() const;
  void _internal_set_pull_propagation_scale(float value);
  public:

  // optional float push_propagation_scale = 4 [default = 8];
  bool has_push_propagation_scale() const;
  private:
  bool _internal_has_push_propagation_scale() const;
  public:
  void clear_push_propagation_scale();
  float push_propagation_scale() const;
  void set_push_propagation_scale(float value);
  private:
  float _internal_push_propagation_scale() const;
  void _internal_set_push_propagation_scale(float value);
  public:

  // optional float pull_bilateral_scale = 5 [default = 0.7];
  bool has_pull_bilateral_scale() const;
  private:
  bool _internal_has_pull_bilateral_scale() const;
  public:
  void clear_pull_bilateral_scale();
  float pull_bilateral_scale() const;
  void set_pull_bilateral_scale(float value);
  private:
  float _internal_pull_bilateral_scale() const;
  void _internal_set_pull_bilateral_scale(float value);
  public:

  GOOGLE_PROTOBUF_EXTENSION_ACCESSORS(PushPullOptions)
  // @@protoc_insertion_point(class_scope:mediapipe.PushPullOptions)
 private:
  class _Internal;

  ::PROTOBUF_NAMESPACE_ID::internal::ExtensionSet _extensions_;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  float push_bilateral_scale_;
  float bilateral_sigma_;
  float pull_propagation_scale_;
  float push_propagation_scale_;
  float pull_bilateral_scale_;
  friend struct ::TableStruct_mediapipe_2futil_2ftracking_2fpush_5fpull_5ffiltering_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// PushPullOptions

// optional float bilateral_sigma = 1 [default = 20];
inline bool PushPullOptions::_internal_has_bilateral_sigma() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool PushPullOptions::has_bilateral_sigma() const {
  return _internal_has_bilateral_sigma();
}
inline void PushPullOptions::clear_bilateral_sigma() {
  bilateral_sigma_ = 20;
  _has_bits_[0] &= ~0x00000002u;
}
inline float PushPullOptions::_internal_bilateral_sigma() const {
  return bilateral_sigma_;
}
inline float PushPullOptions::bilateral_sigma() const {
  // @@protoc_insertion_point(field_get:mediapipe.PushPullOptions.bilateral_sigma)
  return _internal_bilateral_sigma();
}
inline void PushPullOptions::_internal_set_bilateral_sigma(float value) {
  _has_bits_[0] |= 0x00000002u;
  bilateral_sigma_ = value;
}
inline void PushPullOptions::set_bilateral_sigma(float value) {
  _internal_set_bilateral_sigma(value);
  // @@protoc_insertion_point(field_set:mediapipe.PushPullOptions.bilateral_sigma)
}

// optional float pull_propagation_scale = 3 [default = 8];
inline bool PushPullOptions::_internal_has_pull_propagation_scale() const {
  bool value = (_has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool PushPullOptions::has_pull_propagation_scale() const {
  return _internal_has_pull_propagation_scale();
}
inline void PushPullOptions::clear_pull_propagation_scale() {
  pull_propagation_scale_ = 8;
  _has_bits_[0] &= ~0x00000004u;
}
inline float PushPullOptions::_internal_pull_propagation_scale() const {
  return pull_propagation_scale_;
}
inline float PushPullOptions::pull_propagation_scale() const {
  // @@protoc_insertion_point(field_get:mediapipe.PushPullOptions.pull_propagation_scale)
  return _internal_pull_propagation_scale();
}
inline void PushPullOptions::_internal_set_pull_propagation_scale(float value) {
  _has_bits_[0] |= 0x00000004u;
  pull_propagation_scale_ = value;
}
inline void PushPullOptions::set_pull_propagation_scale(float value) {
  _internal_set_pull_propagation_scale(value);
  // @@protoc_insertion_point(field_set:mediapipe.PushPullOptions.pull_propagation_scale)
}

// optional float push_propagation_scale = 4 [default = 8];
inline bool PushPullOptions::_internal_has_push_propagation_scale() const {
  bool value = (_has_bits_[0] & 0x00000008u) != 0;
  return value;
}
inline bool PushPullOptions::has_push_propagation_scale() const {
  return _internal_has_push_propagation_scale();
}
inline void PushPullOptions::clear_push_propagation_scale() {
  push_propagation_scale_ = 8;
  _has_bits_[0] &= ~0x00000008u;
}
inline float PushPullOptions::_internal_push_propagation_scale() const {
  return push_propagation_scale_;
}
inline float PushPullOptions::push_propagation_scale() const {
  // @@protoc_insertion_point(field_get:mediapipe.PushPullOptions.push_propagation_scale)
  return _internal_push_propagation_scale();
}
inline void PushPullOptions::_internal_set_push_propagation_scale(float value) {
  _has_bits_[0] |= 0x00000008u;
  push_propagation_scale_ = value;
}
inline void PushPullOptions::set_push_propagation_scale(float value) {
  _internal_set_push_propagation_scale(value);
  // @@protoc_insertion_point(field_set:mediapipe.PushPullOptions.push_propagation_scale)
}

// optional float pull_bilateral_scale = 5 [default = 0.7];
inline bool PushPullOptions::_internal_has_pull_bilateral_scale() const {
  bool value = (_has_bits_[0] & 0x00000010u) != 0;
  return value;
}
inline bool PushPullOptions::has_pull_bilateral_scale() const {
  return _internal_has_pull_bilateral_scale();
}
inline void PushPullOptions::clear_pull_bilateral_scale() {
  pull_bilateral_scale_ = 0.7f;
  _has_bits_[0] &= ~0x00000010u;
}
inline float PushPullOptions::_internal_pull_bilateral_scale() const {
  return pull_bilateral_scale_;
}
inline float PushPullOptions::pull_bilateral_scale() const {
  // @@protoc_insertion_point(field_get:mediapipe.PushPullOptions.pull_bilateral_scale)
  return _internal_pull_bilateral_scale();
}
inline void PushPullOptions::_internal_set_pull_bilateral_scale(float value) {
  _has_bits_[0] |= 0x00000010u;
  pull_bilateral_scale_ = value;
}
inline void PushPullOptions::set_pull_bilateral_scale(float value) {
  _internal_set_pull_bilateral_scale(value);
  // @@protoc_insertion_point(field_set:mediapipe.PushPullOptions.pull_bilateral_scale)
}

// optional float push_bilateral_scale = 6 [default = 0.9];
inline bool PushPullOptions::_internal_has_push_bilateral_scale() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool PushPullOptions::has_push_bilateral_scale() const {
  return _internal_has_push_bilateral_scale();
}
inline void PushPullOptions::clear_push_bilateral_scale() {
  push_bilateral_scale_ = 0.9f;
  _has_bits_[0] &= ~0x00000001u;
}
inline float PushPullOptions::_internal_push_bilateral_scale() const {
  return push_bilateral_scale_;
}
inline float PushPullOptions::push_bilateral_scale() const {
  // @@protoc_insertion_point(field_get:mediapipe.PushPullOptions.push_bilateral_scale)
  return _internal_push_bilateral_scale();
}
inline void PushPullOptions::_internal_set_push_bilateral_scale(float value) {
  _has_bits_[0] |= 0x00000001u;
  push_bilateral_scale_ = value;
}
inline void PushPullOptions::set_push_bilateral_scale(float value) {
  _internal_set_push_bilateral_scale(value);
  // @@protoc_insertion_point(field_set:mediapipe.PushPullOptions.push_bilateral_scale)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2futil_2ftracking_2fpush_5fpull_5ffiltering_2eproto
