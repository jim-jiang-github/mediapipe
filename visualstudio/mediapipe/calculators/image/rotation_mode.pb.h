// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/image/rotation_mode.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fimage_2frotation_5fmode_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fimage_2frotation_5fmode_2eproto

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
#include <x/google/protobuf/generated_enum_reflection.h>
#include <x/google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <x/google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fcalculators_2fimage_2frotation_5fmode_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fcalculators_2fimage_2frotation_5fmode_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2fimage_2frotation_5fmode_2eproto;
namespace mediapipe {
class RotationMode;
struct RotationModeDefaultTypeInternal;
extern RotationModeDefaultTypeInternal _RotationMode_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::RotationMode* Arena::CreateMaybeMessage<::mediapipe::RotationMode>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

enum RotationMode_Mode : int {
  RotationMode_Mode_UNKNOWN = 0,
  RotationMode_Mode_ROTATION_0 = 1,
  RotationMode_Mode_ROTATION_90 = 2,
  RotationMode_Mode_ROTATION_180 = 3,
  RotationMode_Mode_ROTATION_270 = 4
};
bool RotationMode_Mode_IsValid(int value);
constexpr RotationMode_Mode RotationMode_Mode_Mode_MIN = RotationMode_Mode_UNKNOWN;
constexpr RotationMode_Mode RotationMode_Mode_Mode_MAX = RotationMode_Mode_ROTATION_270;
constexpr int RotationMode_Mode_Mode_ARRAYSIZE = RotationMode_Mode_Mode_MAX + 1;

const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* RotationMode_Mode_descriptor();
template<typename T>
inline const std::string& RotationMode_Mode_Name(T enum_t_value) {
  static_assert(::std::is_same<T, RotationMode_Mode>::value ||
    ::std::is_integral<T>::value,
    "Incorrect type passed to function RotationMode_Mode_Name.");
  return ::PROTOBUF_NAMESPACE_ID::internal::NameOfEnum(
    RotationMode_Mode_descriptor(), enum_t_value);
}
inline bool RotationMode_Mode_Parse(
    ::PROTOBUF_NAMESPACE_ID::ConstStringParam name, RotationMode_Mode* value) {
  return ::PROTOBUF_NAMESPACE_ID::internal::ParseNamedEnum<RotationMode_Mode>(
    RotationMode_Mode_descriptor(), name, value);
}
// ===================================================================

class RotationMode PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.RotationMode) */ {
 public:
  inline RotationMode() : RotationMode(nullptr) {}
  ~RotationMode() override;
  explicit constexpr RotationMode(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  RotationMode(const RotationMode& from);
  RotationMode(RotationMode&& from) noexcept
    : RotationMode() {
    *this = ::std::move(from);
  }

  inline RotationMode& operator=(const RotationMode& from) {
    CopyFrom(from);
    return *this;
  }
  inline RotationMode& operator=(RotationMode&& from) noexcept {
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
  static const RotationMode& default_instance() {
    return *internal_default_instance();
  }
  static inline const RotationMode* internal_default_instance() {
    return reinterpret_cast<const RotationMode*>(
               &_RotationMode_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(RotationMode& a, RotationMode& b) {
    a.Swap(&b);
  }
  inline void Swap(RotationMode* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(RotationMode* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline RotationMode* New() const final {
    return CreateMaybeMessage<RotationMode>(nullptr);
  }

  RotationMode* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<RotationMode>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const RotationMode& from);
  void MergeFrom(const RotationMode& from);
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
  void InternalSwap(RotationMode* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.RotationMode";
  }
  protected:
  explicit RotationMode(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef RotationMode_Mode Mode;
  static constexpr Mode UNKNOWN =
    RotationMode_Mode_UNKNOWN;
  static constexpr Mode ROTATION_0 =
    RotationMode_Mode_ROTATION_0;
  static constexpr Mode ROTATION_90 =
    RotationMode_Mode_ROTATION_90;
  static constexpr Mode ROTATION_180 =
    RotationMode_Mode_ROTATION_180;
  static constexpr Mode ROTATION_270 =
    RotationMode_Mode_ROTATION_270;
  static inline bool Mode_IsValid(int value) {
    return RotationMode_Mode_IsValid(value);
  }
  static constexpr Mode Mode_MIN =
    RotationMode_Mode_Mode_MIN;
  static constexpr Mode Mode_MAX =
    RotationMode_Mode_Mode_MAX;
  static constexpr int Mode_ARRAYSIZE =
    RotationMode_Mode_Mode_ARRAYSIZE;
  static inline const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor*
  Mode_descriptor() {
    return RotationMode_Mode_descriptor();
  }
  template<typename T>
  static inline const std::string& Mode_Name(T enum_t_value) {
    static_assert(::std::is_same<T, Mode>::value ||
      ::std::is_integral<T>::value,
      "Incorrect type passed to function Mode_Name.");
    return RotationMode_Mode_Name(enum_t_value);
  }
  static inline bool Mode_Parse(::PROTOBUF_NAMESPACE_ID::ConstStringParam name,
      Mode* value) {
    return RotationMode_Mode_Parse(name, value);
  }

  // accessors -------------------------------------------------------

  // @@protoc_insertion_point(class_scope:mediapipe.RotationMode)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_mediapipe_2fcalculators_2fimage_2frotation_5fmode_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// RotationMode

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

PROTOBUF_NAMESPACE_OPEN

template <> struct is_proto_enum< ::mediapipe::RotationMode_Mode> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::mediapipe::RotationMode_Mode>() {
  return ::mediapipe::RotationMode_Mode_descriptor();
}

PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)

#include <x/google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fimage_2frotation_5fmode_2eproto
