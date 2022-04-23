// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/image/image_cropping_calculator.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fimage_2fimage_5fcropping_5fcalculator_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fimage_2fimage_5fcropping_5fcalculator_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fcalculators_2fimage_2fimage_5fcropping_5fcalculator_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fcalculators_2fimage_2fimage_5fcropping_5fcalculator_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2fimage_2fimage_5fcropping_5fcalculator_2eproto;
namespace mediapipe {
class ImageCroppingCalculatorOptions;
struct ImageCroppingCalculatorOptionsDefaultTypeInternal;
extern ImageCroppingCalculatorOptionsDefaultTypeInternal _ImageCroppingCalculatorOptions_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::ImageCroppingCalculatorOptions* Arena::CreateMaybeMessage<::mediapipe::ImageCroppingCalculatorOptions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

enum ImageCroppingCalculatorOptions_BorderMode : int {
  ImageCroppingCalculatorOptions_BorderMode_BORDER_UNSPECIFIED = 0,
  ImageCroppingCalculatorOptions_BorderMode_BORDER_ZERO = 1,
  ImageCroppingCalculatorOptions_BorderMode_BORDER_REPLICATE = 2
};
bool ImageCroppingCalculatorOptions_BorderMode_IsValid(int value);
constexpr ImageCroppingCalculatorOptions_BorderMode ImageCroppingCalculatorOptions_BorderMode_BorderMode_MIN = ImageCroppingCalculatorOptions_BorderMode_BORDER_UNSPECIFIED;
constexpr ImageCroppingCalculatorOptions_BorderMode ImageCroppingCalculatorOptions_BorderMode_BorderMode_MAX = ImageCroppingCalculatorOptions_BorderMode_BORDER_REPLICATE;
constexpr int ImageCroppingCalculatorOptions_BorderMode_BorderMode_ARRAYSIZE = ImageCroppingCalculatorOptions_BorderMode_BorderMode_MAX + 1;

const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* ImageCroppingCalculatorOptions_BorderMode_descriptor();
template<typename T>
inline const std::string& ImageCroppingCalculatorOptions_BorderMode_Name(T enum_t_value) {
  static_assert(::std::is_same<T, ImageCroppingCalculatorOptions_BorderMode>::value ||
    ::std::is_integral<T>::value,
    "Incorrect type passed to function ImageCroppingCalculatorOptions_BorderMode_Name.");
  return ::PROTOBUF_NAMESPACE_ID::internal::NameOfEnum(
    ImageCroppingCalculatorOptions_BorderMode_descriptor(), enum_t_value);
}
inline bool ImageCroppingCalculatorOptions_BorderMode_Parse(
    ::PROTOBUF_NAMESPACE_ID::ConstStringParam name, ImageCroppingCalculatorOptions_BorderMode* value) {
  return ::PROTOBUF_NAMESPACE_ID::internal::ParseNamedEnum<ImageCroppingCalculatorOptions_BorderMode>(
    ImageCroppingCalculatorOptions_BorderMode_descriptor(), name, value);
}
// ===================================================================

class ImageCroppingCalculatorOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.ImageCroppingCalculatorOptions) */ {
 public:
  inline ImageCroppingCalculatorOptions() : ImageCroppingCalculatorOptions(nullptr) {}
  ~ImageCroppingCalculatorOptions() override;
  explicit constexpr ImageCroppingCalculatorOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  ImageCroppingCalculatorOptions(const ImageCroppingCalculatorOptions& from);
  ImageCroppingCalculatorOptions(ImageCroppingCalculatorOptions&& from) noexcept
    : ImageCroppingCalculatorOptions() {
    *this = ::std::move(from);
  }

  inline ImageCroppingCalculatorOptions& operator=(const ImageCroppingCalculatorOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline ImageCroppingCalculatorOptions& operator=(ImageCroppingCalculatorOptions&& from) noexcept {
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
  static const ImageCroppingCalculatorOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const ImageCroppingCalculatorOptions* internal_default_instance() {
    return reinterpret_cast<const ImageCroppingCalculatorOptions*>(
               &_ImageCroppingCalculatorOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(ImageCroppingCalculatorOptions& a, ImageCroppingCalculatorOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(ImageCroppingCalculatorOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(ImageCroppingCalculatorOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline ImageCroppingCalculatorOptions* New() const final {
    return CreateMaybeMessage<ImageCroppingCalculatorOptions>(nullptr);
  }

  ImageCroppingCalculatorOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<ImageCroppingCalculatorOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const ImageCroppingCalculatorOptions& from);
  void MergeFrom(const ImageCroppingCalculatorOptions& from);
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
  void InternalSwap(ImageCroppingCalculatorOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.ImageCroppingCalculatorOptions";
  }
  protected:
  explicit ImageCroppingCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef ImageCroppingCalculatorOptions_BorderMode BorderMode;
  static constexpr BorderMode BORDER_UNSPECIFIED =
    ImageCroppingCalculatorOptions_BorderMode_BORDER_UNSPECIFIED;
  static constexpr BorderMode BORDER_ZERO =
    ImageCroppingCalculatorOptions_BorderMode_BORDER_ZERO;
  static constexpr BorderMode BORDER_REPLICATE =
    ImageCroppingCalculatorOptions_BorderMode_BORDER_REPLICATE;
  static inline bool BorderMode_IsValid(int value) {
    return ImageCroppingCalculatorOptions_BorderMode_IsValid(value);
  }
  static constexpr BorderMode BorderMode_MIN =
    ImageCroppingCalculatorOptions_BorderMode_BorderMode_MIN;
  static constexpr BorderMode BorderMode_MAX =
    ImageCroppingCalculatorOptions_BorderMode_BorderMode_MAX;
  static constexpr int BorderMode_ARRAYSIZE =
    ImageCroppingCalculatorOptions_BorderMode_BorderMode_ARRAYSIZE;
  static inline const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor*
  BorderMode_descriptor() {
    return ImageCroppingCalculatorOptions_BorderMode_descriptor();
  }
  template<typename T>
  static inline const std::string& BorderMode_Name(T enum_t_value) {
    static_assert(::std::is_same<T, BorderMode>::value ||
      ::std::is_integral<T>::value,
      "Incorrect type passed to function BorderMode_Name.");
    return ImageCroppingCalculatorOptions_BorderMode_Name(enum_t_value);
  }
  static inline bool BorderMode_Parse(::PROTOBUF_NAMESPACE_ID::ConstStringParam name,
      BorderMode* value) {
    return ImageCroppingCalculatorOptions_BorderMode_Parse(name, value);
  }

  // accessors -------------------------------------------------------

  enum : int {
    kWidthFieldNumber = 1,
    kHeightFieldNumber = 2,
    kRotationFieldNumber = 3,
    kNormWidthFieldNumber = 4,
    kNormHeightFieldNumber = 5,
    kNormCenterXFieldNumber = 6,
    kNormCenterYFieldNumber = 7,
    kOutputMaxWidthFieldNumber = 9,
    kOutputMaxHeightFieldNumber = 10,
    kBorderModeFieldNumber = 8,
  };
  // optional int32 width = 1;
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

  // optional float rotation = 3 [default = 0];
  bool has_rotation() const;
  private:
  bool _internal_has_rotation() const;
  public:
  void clear_rotation();
  float rotation() const;
  void set_rotation(float value);
  private:
  float _internal_rotation() const;
  void _internal_set_rotation(float value);
  public:

  // optional float norm_width = 4;
  bool has_norm_width() const;
  private:
  bool _internal_has_norm_width() const;
  public:
  void clear_norm_width();
  float norm_width() const;
  void set_norm_width(float value);
  private:
  float _internal_norm_width() const;
  void _internal_set_norm_width(float value);
  public:

  // optional float norm_height = 5;
  bool has_norm_height() const;
  private:
  bool _internal_has_norm_height() const;
  public:
  void clear_norm_height();
  float norm_height() const;
  void set_norm_height(float value);
  private:
  float _internal_norm_height() const;
  void _internal_set_norm_height(float value);
  public:

  // optional float norm_center_x = 6 [default = 0];
  bool has_norm_center_x() const;
  private:
  bool _internal_has_norm_center_x() const;
  public:
  void clear_norm_center_x();
  float norm_center_x() const;
  void set_norm_center_x(float value);
  private:
  float _internal_norm_center_x() const;
  void _internal_set_norm_center_x(float value);
  public:

  // optional float norm_center_y = 7 [default = 0];
  bool has_norm_center_y() const;
  private:
  bool _internal_has_norm_center_y() const;
  public:
  void clear_norm_center_y();
  float norm_center_y() const;
  void set_norm_center_y(float value);
  private:
  float _internal_norm_center_y() const;
  void _internal_set_norm_center_y(float value);
  public:

  // optional int32 output_max_width = 9;
  bool has_output_max_width() const;
  private:
  bool _internal_has_output_max_width() const;
  public:
  void clear_output_max_width();
  ::PROTOBUF_NAMESPACE_ID::int32 output_max_width() const;
  void set_output_max_width(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_output_max_width() const;
  void _internal_set_output_max_width(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional int32 output_max_height = 10;
  bool has_output_max_height() const;
  private:
  bool _internal_has_output_max_height() const;
  public:
  void clear_output_max_height();
  ::PROTOBUF_NAMESPACE_ID::int32 output_max_height() const;
  void set_output_max_height(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_output_max_height() const;
  void _internal_set_output_max_height(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional .mediapipe.ImageCroppingCalculatorOptions.BorderMode border_mode = 8 [default = BORDER_ZERO];
  bool has_border_mode() const;
  private:
  bool _internal_has_border_mode() const;
  public:
  void clear_border_mode();
  ::mediapipe::ImageCroppingCalculatorOptions_BorderMode border_mode() const;
  void set_border_mode(::mediapipe::ImageCroppingCalculatorOptions_BorderMode value);
  private:
  ::mediapipe::ImageCroppingCalculatorOptions_BorderMode _internal_border_mode() const;
  void _internal_set_border_mode(::mediapipe::ImageCroppingCalculatorOptions_BorderMode value);
  public:

  static const int kExtFieldNumber = 262466399;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::ImageCroppingCalculatorOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.ImageCroppingCalculatorOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::int32 width_;
  ::PROTOBUF_NAMESPACE_ID::int32 height_;
  float rotation_;
  float norm_width_;
  float norm_height_;
  float norm_center_x_;
  float norm_center_y_;
  ::PROTOBUF_NAMESPACE_ID::int32 output_max_width_;
  ::PROTOBUF_NAMESPACE_ID::int32 output_max_height_;
  int border_mode_;
  friend struct ::TableStruct_mediapipe_2fcalculators_2fimage_2fimage_5fcropping_5fcalculator_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// ImageCroppingCalculatorOptions

// optional int32 width = 1;
inline bool ImageCroppingCalculatorOptions::_internal_has_width() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool ImageCroppingCalculatorOptions::has_width() const {
  return _internal_has_width();
}
inline void ImageCroppingCalculatorOptions::clear_width() {
  width_ = 0;
  _has_bits_[0] &= ~0x00000001u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 ImageCroppingCalculatorOptions::_internal_width() const {
  return width_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 ImageCroppingCalculatorOptions::width() const {
  // @@protoc_insertion_point(field_get:mediapipe.ImageCroppingCalculatorOptions.width)
  return _internal_width();
}
inline void ImageCroppingCalculatorOptions::_internal_set_width(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000001u;
  width_ = value;
}
inline void ImageCroppingCalculatorOptions::set_width(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_width(value);
  // @@protoc_insertion_point(field_set:mediapipe.ImageCroppingCalculatorOptions.width)
}

// optional int32 height = 2;
inline bool ImageCroppingCalculatorOptions::_internal_has_height() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool ImageCroppingCalculatorOptions::has_height() const {
  return _internal_has_height();
}
inline void ImageCroppingCalculatorOptions::clear_height() {
  height_ = 0;
  _has_bits_[0] &= ~0x00000002u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 ImageCroppingCalculatorOptions::_internal_height() const {
  return height_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 ImageCroppingCalculatorOptions::height() const {
  // @@protoc_insertion_point(field_get:mediapipe.ImageCroppingCalculatorOptions.height)
  return _internal_height();
}
inline void ImageCroppingCalculatorOptions::_internal_set_height(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000002u;
  height_ = value;
}
inline void ImageCroppingCalculatorOptions::set_height(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_height(value);
  // @@protoc_insertion_point(field_set:mediapipe.ImageCroppingCalculatorOptions.height)
}

// optional float rotation = 3 [default = 0];
inline bool ImageCroppingCalculatorOptions::_internal_has_rotation() const {
  bool value = (_has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool ImageCroppingCalculatorOptions::has_rotation() const {
  return _internal_has_rotation();
}
inline void ImageCroppingCalculatorOptions::clear_rotation() {
  rotation_ = 0;
  _has_bits_[0] &= ~0x00000004u;
}
inline float ImageCroppingCalculatorOptions::_internal_rotation() const {
  return rotation_;
}
inline float ImageCroppingCalculatorOptions::rotation() const {
  // @@protoc_insertion_point(field_get:mediapipe.ImageCroppingCalculatorOptions.rotation)
  return _internal_rotation();
}
inline void ImageCroppingCalculatorOptions::_internal_set_rotation(float value) {
  _has_bits_[0] |= 0x00000004u;
  rotation_ = value;
}
inline void ImageCroppingCalculatorOptions::set_rotation(float value) {
  _internal_set_rotation(value);
  // @@protoc_insertion_point(field_set:mediapipe.ImageCroppingCalculatorOptions.rotation)
}

// optional float norm_width = 4;
inline bool ImageCroppingCalculatorOptions::_internal_has_norm_width() const {
  bool value = (_has_bits_[0] & 0x00000008u) != 0;
  return value;
}
inline bool ImageCroppingCalculatorOptions::has_norm_width() const {
  return _internal_has_norm_width();
}
inline void ImageCroppingCalculatorOptions::clear_norm_width() {
  norm_width_ = 0;
  _has_bits_[0] &= ~0x00000008u;
}
inline float ImageCroppingCalculatorOptions::_internal_norm_width() const {
  return norm_width_;
}
inline float ImageCroppingCalculatorOptions::norm_width() const {
  // @@protoc_insertion_point(field_get:mediapipe.ImageCroppingCalculatorOptions.norm_width)
  return _internal_norm_width();
}
inline void ImageCroppingCalculatorOptions::_internal_set_norm_width(float value) {
  _has_bits_[0] |= 0x00000008u;
  norm_width_ = value;
}
inline void ImageCroppingCalculatorOptions::set_norm_width(float value) {
  _internal_set_norm_width(value);
  // @@protoc_insertion_point(field_set:mediapipe.ImageCroppingCalculatorOptions.norm_width)
}

// optional float norm_height = 5;
inline bool ImageCroppingCalculatorOptions::_internal_has_norm_height() const {
  bool value = (_has_bits_[0] & 0x00000010u) != 0;
  return value;
}
inline bool ImageCroppingCalculatorOptions::has_norm_height() const {
  return _internal_has_norm_height();
}
inline void ImageCroppingCalculatorOptions::clear_norm_height() {
  norm_height_ = 0;
  _has_bits_[0] &= ~0x00000010u;
}
inline float ImageCroppingCalculatorOptions::_internal_norm_height() const {
  return norm_height_;
}
inline float ImageCroppingCalculatorOptions::norm_height() const {
  // @@protoc_insertion_point(field_get:mediapipe.ImageCroppingCalculatorOptions.norm_height)
  return _internal_norm_height();
}
inline void ImageCroppingCalculatorOptions::_internal_set_norm_height(float value) {
  _has_bits_[0] |= 0x00000010u;
  norm_height_ = value;
}
inline void ImageCroppingCalculatorOptions::set_norm_height(float value) {
  _internal_set_norm_height(value);
  // @@protoc_insertion_point(field_set:mediapipe.ImageCroppingCalculatorOptions.norm_height)
}

// optional float norm_center_x = 6 [default = 0];
inline bool ImageCroppingCalculatorOptions::_internal_has_norm_center_x() const {
  bool value = (_has_bits_[0] & 0x00000020u) != 0;
  return value;
}
inline bool ImageCroppingCalculatorOptions::has_norm_center_x() const {
  return _internal_has_norm_center_x();
}
inline void ImageCroppingCalculatorOptions::clear_norm_center_x() {
  norm_center_x_ = 0;
  _has_bits_[0] &= ~0x00000020u;
}
inline float ImageCroppingCalculatorOptions::_internal_norm_center_x() const {
  return norm_center_x_;
}
inline float ImageCroppingCalculatorOptions::norm_center_x() const {
  // @@protoc_insertion_point(field_get:mediapipe.ImageCroppingCalculatorOptions.norm_center_x)
  return _internal_norm_center_x();
}
inline void ImageCroppingCalculatorOptions::_internal_set_norm_center_x(float value) {
  _has_bits_[0] |= 0x00000020u;
  norm_center_x_ = value;
}
inline void ImageCroppingCalculatorOptions::set_norm_center_x(float value) {
  _internal_set_norm_center_x(value);
  // @@protoc_insertion_point(field_set:mediapipe.ImageCroppingCalculatorOptions.norm_center_x)
}

// optional float norm_center_y = 7 [default = 0];
inline bool ImageCroppingCalculatorOptions::_internal_has_norm_center_y() const {
  bool value = (_has_bits_[0] & 0x00000040u) != 0;
  return value;
}
inline bool ImageCroppingCalculatorOptions::has_norm_center_y() const {
  return _internal_has_norm_center_y();
}
inline void ImageCroppingCalculatorOptions::clear_norm_center_y() {
  norm_center_y_ = 0;
  _has_bits_[0] &= ~0x00000040u;
}
inline float ImageCroppingCalculatorOptions::_internal_norm_center_y() const {
  return norm_center_y_;
}
inline float ImageCroppingCalculatorOptions::norm_center_y() const {
  // @@protoc_insertion_point(field_get:mediapipe.ImageCroppingCalculatorOptions.norm_center_y)
  return _internal_norm_center_y();
}
inline void ImageCroppingCalculatorOptions::_internal_set_norm_center_y(float value) {
  _has_bits_[0] |= 0x00000040u;
  norm_center_y_ = value;
}
inline void ImageCroppingCalculatorOptions::set_norm_center_y(float value) {
  _internal_set_norm_center_y(value);
  // @@protoc_insertion_point(field_set:mediapipe.ImageCroppingCalculatorOptions.norm_center_y)
}

// optional .mediapipe.ImageCroppingCalculatorOptions.BorderMode border_mode = 8 [default = BORDER_ZERO];
inline bool ImageCroppingCalculatorOptions::_internal_has_border_mode() const {
  bool value = (_has_bits_[0] & 0x00000200u) != 0;
  return value;
}
inline bool ImageCroppingCalculatorOptions::has_border_mode() const {
  return _internal_has_border_mode();
}
inline void ImageCroppingCalculatorOptions::clear_border_mode() {
  border_mode_ = 1;
  _has_bits_[0] &= ~0x00000200u;
}
inline ::mediapipe::ImageCroppingCalculatorOptions_BorderMode ImageCroppingCalculatorOptions::_internal_border_mode() const {
  return static_cast< ::mediapipe::ImageCroppingCalculatorOptions_BorderMode >(border_mode_);
}
inline ::mediapipe::ImageCroppingCalculatorOptions_BorderMode ImageCroppingCalculatorOptions::border_mode() const {
  // @@protoc_insertion_point(field_get:mediapipe.ImageCroppingCalculatorOptions.border_mode)
  return _internal_border_mode();
}
inline void ImageCroppingCalculatorOptions::_internal_set_border_mode(::mediapipe::ImageCroppingCalculatorOptions_BorderMode value) {
  assert(::mediapipe::ImageCroppingCalculatorOptions_BorderMode_IsValid(value));
  _has_bits_[0] |= 0x00000200u;
  border_mode_ = value;
}
inline void ImageCroppingCalculatorOptions::set_border_mode(::mediapipe::ImageCroppingCalculatorOptions_BorderMode value) {
  _internal_set_border_mode(value);
  // @@protoc_insertion_point(field_set:mediapipe.ImageCroppingCalculatorOptions.border_mode)
}

// optional int32 output_max_width = 9;
inline bool ImageCroppingCalculatorOptions::_internal_has_output_max_width() const {
  bool value = (_has_bits_[0] & 0x00000080u) != 0;
  return value;
}
inline bool ImageCroppingCalculatorOptions::has_output_max_width() const {
  return _internal_has_output_max_width();
}
inline void ImageCroppingCalculatorOptions::clear_output_max_width() {
  output_max_width_ = 0;
  _has_bits_[0] &= ~0x00000080u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 ImageCroppingCalculatorOptions::_internal_output_max_width() const {
  return output_max_width_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 ImageCroppingCalculatorOptions::output_max_width() const {
  // @@protoc_insertion_point(field_get:mediapipe.ImageCroppingCalculatorOptions.output_max_width)
  return _internal_output_max_width();
}
inline void ImageCroppingCalculatorOptions::_internal_set_output_max_width(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000080u;
  output_max_width_ = value;
}
inline void ImageCroppingCalculatorOptions::set_output_max_width(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_output_max_width(value);
  // @@protoc_insertion_point(field_set:mediapipe.ImageCroppingCalculatorOptions.output_max_width)
}

// optional int32 output_max_height = 10;
inline bool ImageCroppingCalculatorOptions::_internal_has_output_max_height() const {
  bool value = (_has_bits_[0] & 0x00000100u) != 0;
  return value;
}
inline bool ImageCroppingCalculatorOptions::has_output_max_height() const {
  return _internal_has_output_max_height();
}
inline void ImageCroppingCalculatorOptions::clear_output_max_height() {
  output_max_height_ = 0;
  _has_bits_[0] &= ~0x00000100u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 ImageCroppingCalculatorOptions::_internal_output_max_height() const {
  return output_max_height_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 ImageCroppingCalculatorOptions::output_max_height() const {
  // @@protoc_insertion_point(field_get:mediapipe.ImageCroppingCalculatorOptions.output_max_height)
  return _internal_output_max_height();
}
inline void ImageCroppingCalculatorOptions::_internal_set_output_max_height(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000100u;
  output_max_height_ = value;
}
inline void ImageCroppingCalculatorOptions::set_output_max_height(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_output_max_height(value);
  // @@protoc_insertion_point(field_set:mediapipe.ImageCroppingCalculatorOptions.output_max_height)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

PROTOBUF_NAMESPACE_OPEN

template <> struct is_proto_enum< ::mediapipe::ImageCroppingCalculatorOptions_BorderMode> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::mediapipe::ImageCroppingCalculatorOptions_BorderMode>() {
  return ::mediapipe::ImageCroppingCalculatorOptions_BorderMode_descriptor();
}

PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fimage_2fimage_5fcropping_5fcalculator_2eproto
