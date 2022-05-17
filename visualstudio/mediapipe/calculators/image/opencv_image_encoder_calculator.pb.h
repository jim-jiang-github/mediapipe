// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/image/opencv_image_encoder_calculator.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fimage_2fopencv_5fimage_5fencoder_5fcalculator_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fimage_2fopencv_5fimage_5fencoder_5fcalculator_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fcalculators_2fimage_2fopencv_5fimage_5fencoder_5fcalculator_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fcalculators_2fimage_2fopencv_5fimage_5fencoder_5fcalculator_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[2]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2fimage_2fopencv_5fimage_5fencoder_5fcalculator_2eproto;
namespace mediapipe {
class OpenCvImageEncoderCalculatorOptions;
struct OpenCvImageEncoderCalculatorOptionsDefaultTypeInternal;
extern OpenCvImageEncoderCalculatorOptionsDefaultTypeInternal _OpenCvImageEncoderCalculatorOptions_default_instance_;
class OpenCvImageEncoderCalculatorResults;
struct OpenCvImageEncoderCalculatorResultsDefaultTypeInternal;
extern OpenCvImageEncoderCalculatorResultsDefaultTypeInternal _OpenCvImageEncoderCalculatorResults_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::OpenCvImageEncoderCalculatorOptions* Arena::CreateMaybeMessage<::mediapipe::OpenCvImageEncoderCalculatorOptions>(Arena*);
template<> ::mediapipe::OpenCvImageEncoderCalculatorResults* Arena::CreateMaybeMessage<::mediapipe::OpenCvImageEncoderCalculatorResults>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

enum OpenCvImageEncoderCalculatorResults_ColorSpace : int {
  OpenCvImageEncoderCalculatorResults_ColorSpace_UNKNOWN = 0,
  OpenCvImageEncoderCalculatorResults_ColorSpace_GRAYSCALE = 1,
  OpenCvImageEncoderCalculatorResults_ColorSpace_RGB = 2
};
bool OpenCvImageEncoderCalculatorResults_ColorSpace_IsValid(int value);
constexpr OpenCvImageEncoderCalculatorResults_ColorSpace OpenCvImageEncoderCalculatorResults_ColorSpace_ColorSpace_MIN = OpenCvImageEncoderCalculatorResults_ColorSpace_UNKNOWN;
constexpr OpenCvImageEncoderCalculatorResults_ColorSpace OpenCvImageEncoderCalculatorResults_ColorSpace_ColorSpace_MAX = OpenCvImageEncoderCalculatorResults_ColorSpace_RGB;
constexpr int OpenCvImageEncoderCalculatorResults_ColorSpace_ColorSpace_ARRAYSIZE = OpenCvImageEncoderCalculatorResults_ColorSpace_ColorSpace_MAX + 1;

const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* OpenCvImageEncoderCalculatorResults_ColorSpace_descriptor();
template<typename T>
inline const std::string& OpenCvImageEncoderCalculatorResults_ColorSpace_Name(T enum_t_value) {
  static_assert(::std::is_same<T, OpenCvImageEncoderCalculatorResults_ColorSpace>::value ||
    ::std::is_integral<T>::value,
    "Incorrect type passed to function OpenCvImageEncoderCalculatorResults_ColorSpace_Name.");
  return ::PROTOBUF_NAMESPACE_ID::internal::NameOfEnum(
    OpenCvImageEncoderCalculatorResults_ColorSpace_descriptor(), enum_t_value);
}
inline bool OpenCvImageEncoderCalculatorResults_ColorSpace_Parse(
    ::PROTOBUF_NAMESPACE_ID::ConstStringParam name, OpenCvImageEncoderCalculatorResults_ColorSpace* value) {
  return ::PROTOBUF_NAMESPACE_ID::internal::ParseNamedEnum<OpenCvImageEncoderCalculatorResults_ColorSpace>(
    OpenCvImageEncoderCalculatorResults_ColorSpace_descriptor(), name, value);
}
// ===================================================================

class OpenCvImageEncoderCalculatorOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.OpenCvImageEncoderCalculatorOptions) */ {
 public:
  inline OpenCvImageEncoderCalculatorOptions() : OpenCvImageEncoderCalculatorOptions(nullptr) {}
  ~OpenCvImageEncoderCalculatorOptions() override;
  explicit constexpr OpenCvImageEncoderCalculatorOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  OpenCvImageEncoderCalculatorOptions(const OpenCvImageEncoderCalculatorOptions& from);
  OpenCvImageEncoderCalculatorOptions(OpenCvImageEncoderCalculatorOptions&& from) noexcept
    : OpenCvImageEncoderCalculatorOptions() {
    *this = ::std::move(from);
  }

  inline OpenCvImageEncoderCalculatorOptions& operator=(const OpenCvImageEncoderCalculatorOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline OpenCvImageEncoderCalculatorOptions& operator=(OpenCvImageEncoderCalculatorOptions&& from) noexcept {
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
  static const OpenCvImageEncoderCalculatorOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const OpenCvImageEncoderCalculatorOptions* internal_default_instance() {
    return reinterpret_cast<const OpenCvImageEncoderCalculatorOptions*>(
               &_OpenCvImageEncoderCalculatorOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(OpenCvImageEncoderCalculatorOptions& a, OpenCvImageEncoderCalculatorOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(OpenCvImageEncoderCalculatorOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(OpenCvImageEncoderCalculatorOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline OpenCvImageEncoderCalculatorOptions* New() const final {
    return CreateMaybeMessage<OpenCvImageEncoderCalculatorOptions>(nullptr);
  }

  OpenCvImageEncoderCalculatorOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<OpenCvImageEncoderCalculatorOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const OpenCvImageEncoderCalculatorOptions& from);
  void MergeFrom(const OpenCvImageEncoderCalculatorOptions& from);
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
  void InternalSwap(OpenCvImageEncoderCalculatorOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.OpenCvImageEncoderCalculatorOptions";
  }
  protected:
  explicit OpenCvImageEncoderCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kQualityFieldNumber = 1,
  };
  // optional int32 quality = 1;
  bool has_quality() const;
  private:
  bool _internal_has_quality() const;
  public:
  void clear_quality();
  ::PROTOBUF_NAMESPACE_ID::int32 quality() const;
  void set_quality(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_quality() const;
  void _internal_set_quality(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  static const int kExtFieldNumber = 227563646;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::OpenCvImageEncoderCalculatorOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.OpenCvImageEncoderCalculatorOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::int32 quality_;
  friend struct ::TableStruct_mediapipe_2fcalculators_2fimage_2fopencv_5fimage_5fencoder_5fcalculator_2eproto;
};
// -------------------------------------------------------------------

class OpenCvImageEncoderCalculatorResults PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.OpenCvImageEncoderCalculatorResults) */ {
 public:
  inline OpenCvImageEncoderCalculatorResults() : OpenCvImageEncoderCalculatorResults(nullptr) {}
  ~OpenCvImageEncoderCalculatorResults() override;
  explicit constexpr OpenCvImageEncoderCalculatorResults(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  OpenCvImageEncoderCalculatorResults(const OpenCvImageEncoderCalculatorResults& from);
  OpenCvImageEncoderCalculatorResults(OpenCvImageEncoderCalculatorResults&& from) noexcept
    : OpenCvImageEncoderCalculatorResults() {
    *this = ::std::move(from);
  }

  inline OpenCvImageEncoderCalculatorResults& operator=(const OpenCvImageEncoderCalculatorResults& from) {
    CopyFrom(from);
    return *this;
  }
  inline OpenCvImageEncoderCalculatorResults& operator=(OpenCvImageEncoderCalculatorResults&& from) noexcept {
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
  static const OpenCvImageEncoderCalculatorResults& default_instance() {
    return *internal_default_instance();
  }
  static inline const OpenCvImageEncoderCalculatorResults* internal_default_instance() {
    return reinterpret_cast<const OpenCvImageEncoderCalculatorResults*>(
               &_OpenCvImageEncoderCalculatorResults_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(OpenCvImageEncoderCalculatorResults& a, OpenCvImageEncoderCalculatorResults& b) {
    a.Swap(&b);
  }
  inline void Swap(OpenCvImageEncoderCalculatorResults* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(OpenCvImageEncoderCalculatorResults* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline OpenCvImageEncoderCalculatorResults* New() const final {
    return CreateMaybeMessage<OpenCvImageEncoderCalculatorResults>(nullptr);
  }

  OpenCvImageEncoderCalculatorResults* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<OpenCvImageEncoderCalculatorResults>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const OpenCvImageEncoderCalculatorResults& from);
  void MergeFrom(const OpenCvImageEncoderCalculatorResults& from);
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
  void InternalSwap(OpenCvImageEncoderCalculatorResults* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.OpenCvImageEncoderCalculatorResults";
  }
  protected:
  explicit OpenCvImageEncoderCalculatorResults(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef OpenCvImageEncoderCalculatorResults_ColorSpace ColorSpace;
  static constexpr ColorSpace UNKNOWN =
    OpenCvImageEncoderCalculatorResults_ColorSpace_UNKNOWN;
  static constexpr ColorSpace GRAYSCALE =
    OpenCvImageEncoderCalculatorResults_ColorSpace_GRAYSCALE;
  static constexpr ColorSpace RGB =
    OpenCvImageEncoderCalculatorResults_ColorSpace_RGB;
  static inline bool ColorSpace_IsValid(int value) {
    return OpenCvImageEncoderCalculatorResults_ColorSpace_IsValid(value);
  }
  static constexpr ColorSpace ColorSpace_MIN =
    OpenCvImageEncoderCalculatorResults_ColorSpace_ColorSpace_MIN;
  static constexpr ColorSpace ColorSpace_MAX =
    OpenCvImageEncoderCalculatorResults_ColorSpace_ColorSpace_MAX;
  static constexpr int ColorSpace_ARRAYSIZE =
    OpenCvImageEncoderCalculatorResults_ColorSpace_ColorSpace_ARRAYSIZE;
  static inline const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor*
  ColorSpace_descriptor() {
    return OpenCvImageEncoderCalculatorResults_ColorSpace_descriptor();
  }
  template<typename T>
  static inline const std::string& ColorSpace_Name(T enum_t_value) {
    static_assert(::std::is_same<T, ColorSpace>::value ||
      ::std::is_integral<T>::value,
      "Incorrect type passed to function ColorSpace_Name.");
    return OpenCvImageEncoderCalculatorResults_ColorSpace_Name(enum_t_value);
  }
  static inline bool ColorSpace_Parse(::PROTOBUF_NAMESPACE_ID::ConstStringParam name,
      ColorSpace* value) {
    return OpenCvImageEncoderCalculatorResults_ColorSpace_Parse(name, value);
  }

  // accessors -------------------------------------------------------

  enum : int {
    kEncodedImageFieldNumber = 1,
    kHeightFieldNumber = 2,
    kWidthFieldNumber = 3,
    kColorspaceFieldNumber = 4,
  };
  // optional bytes encoded_image = 1;
  bool has_encoded_image() const;
  private:
  bool _internal_has_encoded_image() const;
  public:
  void clear_encoded_image();
  const std::string& encoded_image() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_encoded_image(ArgT0&& arg0, ArgT... args);
  std::string* mutable_encoded_image();
  std::string* release_encoded_image();
  void set_allocated_encoded_image(std::string* encoded_image);
  private:
  const std::string& _internal_encoded_image() const;
  void _internal_set_encoded_image(const std::string& value);
  std::string* _internal_mutable_encoded_image();
  public:

  // optional int32 height = 2;
  bool has_height() const;
  private:
  bool _internal_has_height() const;
  public:
  void clear_height();
  ::PROTOBUF_NAMESPACE_ID::int32 height() const;
  void set_height(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_height() const;
  void _internal_set_height(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional int32 width = 3;
  bool has_width() const;
  private:
  bool _internal_has_width() const;
  public:
  void clear_width();
  ::PROTOBUF_NAMESPACE_ID::int32 width() const;
  void set_width(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_width() const;
  void _internal_set_width(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional .mediapipe.OpenCvImageEncoderCalculatorResults.ColorSpace colorspace = 4;
  bool has_colorspace() const;
  private:
  bool _internal_has_colorspace() const;
  public:
  void clear_colorspace();
  ::mediapipe::OpenCvImageEncoderCalculatorResults_ColorSpace colorspace() const;
  void set_colorspace(::mediapipe::OpenCvImageEncoderCalculatorResults_ColorSpace value);
  private:
  ::mediapipe::OpenCvImageEncoderCalculatorResults_ColorSpace _internal_colorspace() const;
  void _internal_set_colorspace(::mediapipe::OpenCvImageEncoderCalculatorResults_ColorSpace value);
  public:

  // @@protoc_insertion_point(class_scope:mediapipe.OpenCvImageEncoderCalculatorResults)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr encoded_image_;
  ::PROTOBUF_NAMESPACE_ID::int32 height_;
  ::PROTOBUF_NAMESPACE_ID::int32 width_;
  int colorspace_;
  friend struct ::TableStruct_mediapipe_2fcalculators_2fimage_2fopencv_5fimage_5fencoder_5fcalculator_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// OpenCvImageEncoderCalculatorOptions

// optional int32 quality = 1;
inline bool OpenCvImageEncoderCalculatorOptions::_internal_has_quality() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool OpenCvImageEncoderCalculatorOptions::has_quality() const {
  return _internal_has_quality();
}
inline void OpenCvImageEncoderCalculatorOptions::clear_quality() {
  quality_ = 0;
  _has_bits_[0] &= ~0x00000001u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 OpenCvImageEncoderCalculatorOptions::_internal_quality() const {
  return quality_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 OpenCvImageEncoderCalculatorOptions::quality() const {
  // @@protoc_insertion_point(field_get:mediapipe.OpenCvImageEncoderCalculatorOptions.quality)
  return _internal_quality();
}
inline void OpenCvImageEncoderCalculatorOptions::_internal_set_quality(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000001u;
  quality_ = value;
}
inline void OpenCvImageEncoderCalculatorOptions::set_quality(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_quality(value);
  // @@protoc_insertion_point(field_set:mediapipe.OpenCvImageEncoderCalculatorOptions.quality)
}

// -------------------------------------------------------------------

// OpenCvImageEncoderCalculatorResults

// optional bytes encoded_image = 1;
inline bool OpenCvImageEncoderCalculatorResults::_internal_has_encoded_image() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool OpenCvImageEncoderCalculatorResults::has_encoded_image() const {
  return _internal_has_encoded_image();
}
inline void OpenCvImageEncoderCalculatorResults::clear_encoded_image() {
  encoded_image_.ClearToEmpty();
  _has_bits_[0] &= ~0x00000001u;
}
inline const std::string& OpenCvImageEncoderCalculatorResults::encoded_image() const {
  // @@protoc_insertion_point(field_get:mediapipe.OpenCvImageEncoderCalculatorResults.encoded_image)
  return _internal_encoded_image();
}
template <typename ArgT0, typename... ArgT>
PROTOBUF_ALWAYS_INLINE
inline void OpenCvImageEncoderCalculatorResults::set_encoded_image(ArgT0&& arg0, ArgT... args) {
 _has_bits_[0] |= 0x00000001u;
 encoded_image_.SetBytes(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArena());
  // @@protoc_insertion_point(field_set:mediapipe.OpenCvImageEncoderCalculatorResults.encoded_image)
}
inline std::string* OpenCvImageEncoderCalculatorResults::mutable_encoded_image() {
  // @@protoc_insertion_point(field_mutable:mediapipe.OpenCvImageEncoderCalculatorResults.encoded_image)
  return _internal_mutable_encoded_image();
}
inline const std::string& OpenCvImageEncoderCalculatorResults::_internal_encoded_image() const {
  return encoded_image_.Get();
}
inline void OpenCvImageEncoderCalculatorResults::_internal_set_encoded_image(const std::string& value) {
  _has_bits_[0] |= 0x00000001u;
  encoded_image_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArena());
}
inline std::string* OpenCvImageEncoderCalculatorResults::_internal_mutable_encoded_image() {
  _has_bits_[0] |= 0x00000001u;
  return encoded_image_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArena());
}
inline std::string* OpenCvImageEncoderCalculatorResults::release_encoded_image() {
  // @@protoc_insertion_point(field_release:mediapipe.OpenCvImageEncoderCalculatorResults.encoded_image)
  if (!_internal_has_encoded_image()) {
    return nullptr;
  }
  _has_bits_[0] &= ~0x00000001u;
  return encoded_image_.ReleaseNonDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
}
inline void OpenCvImageEncoderCalculatorResults::set_allocated_encoded_image(std::string* encoded_image) {
  if (encoded_image != nullptr) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  encoded_image_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), encoded_image,
      GetArena());
  // @@protoc_insertion_point(field_set_allocated:mediapipe.OpenCvImageEncoderCalculatorResults.encoded_image)
}

// optional int32 height = 2;
inline bool OpenCvImageEncoderCalculatorResults::_internal_has_height() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool OpenCvImageEncoderCalculatorResults::has_height() const {
  return _internal_has_height();
}
inline void OpenCvImageEncoderCalculatorResults::clear_height() {
  height_ = 0;
  _has_bits_[0] &= ~0x00000002u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 OpenCvImageEncoderCalculatorResults::_internal_height() const {
  return height_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 OpenCvImageEncoderCalculatorResults::height() const {
  // @@protoc_insertion_point(field_get:mediapipe.OpenCvImageEncoderCalculatorResults.height)
  return _internal_height();
}
inline void OpenCvImageEncoderCalculatorResults::_internal_set_height(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000002u;
  height_ = value;
}
inline void OpenCvImageEncoderCalculatorResults::set_height(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_height(value);
  // @@protoc_insertion_point(field_set:mediapipe.OpenCvImageEncoderCalculatorResults.height)
}

// optional int32 width = 3;
inline bool OpenCvImageEncoderCalculatorResults::_internal_has_width() const {
  bool value = (_has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool OpenCvImageEncoderCalculatorResults::has_width() const {
  return _internal_has_width();
}
inline void OpenCvImageEncoderCalculatorResults::clear_width() {
  width_ = 0;
  _has_bits_[0] &= ~0x00000004u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 OpenCvImageEncoderCalculatorResults::_internal_width() const {
  return width_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 OpenCvImageEncoderCalculatorResults::width() const {
  // @@protoc_insertion_point(field_get:mediapipe.OpenCvImageEncoderCalculatorResults.width)
  return _internal_width();
}
inline void OpenCvImageEncoderCalculatorResults::_internal_set_width(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000004u;
  width_ = value;
}
inline void OpenCvImageEncoderCalculatorResults::set_width(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_width(value);
  // @@protoc_insertion_point(field_set:mediapipe.OpenCvImageEncoderCalculatorResults.width)
}

// optional .mediapipe.OpenCvImageEncoderCalculatorResults.ColorSpace colorspace = 4;
inline bool OpenCvImageEncoderCalculatorResults::_internal_has_colorspace() const {
  bool value = (_has_bits_[0] & 0x00000008u) != 0;
  return value;
}
inline bool OpenCvImageEncoderCalculatorResults::has_colorspace() const {
  return _internal_has_colorspace();
}
inline void OpenCvImageEncoderCalculatorResults::clear_colorspace() {
  colorspace_ = 0;
  _has_bits_[0] &= ~0x00000008u;
}
inline ::mediapipe::OpenCvImageEncoderCalculatorResults_ColorSpace OpenCvImageEncoderCalculatorResults::_internal_colorspace() const {
  return static_cast< ::mediapipe::OpenCvImageEncoderCalculatorResults_ColorSpace >(colorspace_);
}
inline ::mediapipe::OpenCvImageEncoderCalculatorResults_ColorSpace OpenCvImageEncoderCalculatorResults::colorspace() const {
  // @@protoc_insertion_point(field_get:mediapipe.OpenCvImageEncoderCalculatorResults.colorspace)
  return _internal_colorspace();
}
inline void OpenCvImageEncoderCalculatorResults::_internal_set_colorspace(::mediapipe::OpenCvImageEncoderCalculatorResults_ColorSpace value) {
  assert(::mediapipe::OpenCvImageEncoderCalculatorResults_ColorSpace_IsValid(value));
  _has_bits_[0] |= 0x00000008u;
  colorspace_ = value;
}
inline void OpenCvImageEncoderCalculatorResults::set_colorspace(::mediapipe::OpenCvImageEncoderCalculatorResults_ColorSpace value) {
  _internal_set_colorspace(value);
  // @@protoc_insertion_point(field_set:mediapipe.OpenCvImageEncoderCalculatorResults.colorspace)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

PROTOBUF_NAMESPACE_OPEN

template <> struct is_proto_enum< ::mediapipe::OpenCvImageEncoderCalculatorResults_ColorSpace> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::mediapipe::OpenCvImageEncoderCalculatorResults_ColorSpace>() {
  return ::mediapipe::OpenCvImageEncoderCalculatorResults_ColorSpace_descriptor();
}

PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fimage_2fopencv_5fimage_5fencoder_5fcalculator_2eproto
