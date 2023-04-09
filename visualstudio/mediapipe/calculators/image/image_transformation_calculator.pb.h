// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/image/image_transformation_calculator.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fimage_2fimage_5ftransformation_5fcalculator_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fimage_2fimage_5ftransformation_5fcalculator_2eproto

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
#include "mediapipe/calculators/image/rotation_mode.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/gpu/scale_mode.pb.h"
// @@protoc_insertion_point(includes)
#include <x/google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fcalculators_2fimage_2fimage_5ftransformation_5fcalculator_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fcalculators_2fimage_2fimage_5ftransformation_5fcalculator_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2fimage_2fimage_5ftransformation_5fcalculator_2eproto;
namespace mediapipe {
class ImageTransformationCalculatorOptions;
struct ImageTransformationCalculatorOptionsDefaultTypeInternal;
extern ImageTransformationCalculatorOptionsDefaultTypeInternal _ImageTransformationCalculatorOptions_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::ImageTransformationCalculatorOptions* Arena::CreateMaybeMessage<::mediapipe::ImageTransformationCalculatorOptions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class ImageTransformationCalculatorOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.ImageTransformationCalculatorOptions) */ {
 public:
  inline ImageTransformationCalculatorOptions() : ImageTransformationCalculatorOptions(nullptr) {}
  ~ImageTransformationCalculatorOptions() override;
  explicit constexpr ImageTransformationCalculatorOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  ImageTransformationCalculatorOptions(const ImageTransformationCalculatorOptions& from);
  ImageTransformationCalculatorOptions(ImageTransformationCalculatorOptions&& from) noexcept
    : ImageTransformationCalculatorOptions() {
    *this = ::std::move(from);
  }

  inline ImageTransformationCalculatorOptions& operator=(const ImageTransformationCalculatorOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline ImageTransformationCalculatorOptions& operator=(ImageTransformationCalculatorOptions&& from) noexcept {
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
  static const ImageTransformationCalculatorOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const ImageTransformationCalculatorOptions* internal_default_instance() {
    return reinterpret_cast<const ImageTransformationCalculatorOptions*>(
               &_ImageTransformationCalculatorOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(ImageTransformationCalculatorOptions& a, ImageTransformationCalculatorOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(ImageTransformationCalculatorOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(ImageTransformationCalculatorOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline ImageTransformationCalculatorOptions* New() const final {
    return CreateMaybeMessage<ImageTransformationCalculatorOptions>(nullptr);
  }

  ImageTransformationCalculatorOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<ImageTransformationCalculatorOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const ImageTransformationCalculatorOptions& from);
  void MergeFrom(const ImageTransformationCalculatorOptions& from);
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
  void InternalSwap(ImageTransformationCalculatorOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.ImageTransformationCalculatorOptions";
  }
  protected:
  explicit ImageTransformationCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kOutputWidthFieldNumber = 1,
    kOutputHeightFieldNumber = 2,
    kRotationModeFieldNumber = 3,
    kFlipVerticallyFieldNumber = 4,
    kFlipHorizontallyFieldNumber = 5,
    kScaleModeFieldNumber = 6,
    kConstantPaddingFieldNumber = 7,
  };
  // optional int32 output_width = 1 [default = 0];
  bool has_output_width() const;
  private:
  bool _internal_has_output_width() const;
  public:
  void clear_output_width();
  ::PROTOBUF_NAMESPACE_ID::int32 output_width() const;
  void set_output_width(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_output_width() const;
  void _internal_set_output_width(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional int32 output_height = 2 [default = 0];
  bool has_output_height() const;
  private:
  bool _internal_has_output_height() const;
  public:
  void clear_output_height();
  ::PROTOBUF_NAMESPACE_ID::int32 output_height() const;
  void set_output_height(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_output_height() const;
  void _internal_set_output_height(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional .mediapipe.RotationMode.Mode rotation_mode = 3;
  bool has_rotation_mode() const;
  private:
  bool _internal_has_rotation_mode() const;
  public:
  void clear_rotation_mode();
  ::mediapipe::RotationMode_Mode rotation_mode() const;
  void set_rotation_mode(::mediapipe::RotationMode_Mode value);
  private:
  ::mediapipe::RotationMode_Mode _internal_rotation_mode() const;
  void _internal_set_rotation_mode(::mediapipe::RotationMode_Mode value);
  public:

  // optional bool flip_vertically = 4 [default = false];
  bool has_flip_vertically() const;
  private:
  bool _internal_has_flip_vertically() const;
  public:
  void clear_flip_vertically();
  bool flip_vertically() const;
  void set_flip_vertically(bool value);
  private:
  bool _internal_flip_vertically() const;
  void _internal_set_flip_vertically(bool value);
  public:

  // optional bool flip_horizontally = 5 [default = false];
  bool has_flip_horizontally() const;
  private:
  bool _internal_has_flip_horizontally() const;
  public:
  void clear_flip_horizontally();
  bool flip_horizontally() const;
  void set_flip_horizontally(bool value);
  private:
  bool _internal_flip_horizontally() const;
  void _internal_set_flip_horizontally(bool value);
  public:

  // optional .mediapipe.ScaleMode.Mode scale_mode = 6;
  bool has_scale_mode() const;
  private:
  bool _internal_has_scale_mode() const;
  public:
  void clear_scale_mode();
  ::mediapipe::ScaleMode_Mode scale_mode() const;
  void set_scale_mode(::mediapipe::ScaleMode_Mode value);
  private:
  ::mediapipe::ScaleMode_Mode _internal_scale_mode() const;
  void _internal_set_scale_mode(::mediapipe::ScaleMode_Mode value);
  public:

  // optional bool constant_padding = 7 [default = true];
  bool has_constant_padding() const;
  private:
  bool _internal_has_constant_padding() const;
  public:
  void clear_constant_padding();
  bool constant_padding() const;
  void set_constant_padding(bool value);
  private:
  bool _internal_constant_padding() const;
  void _internal_set_constant_padding(bool value);
  public:

  static const int kExtFieldNumber = 251952830;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::ImageTransformationCalculatorOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.ImageTransformationCalculatorOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::int32 output_width_;
  ::PROTOBUF_NAMESPACE_ID::int32 output_height_;
  int rotation_mode_;
  bool flip_vertically_;
  bool flip_horizontally_;
  int scale_mode_;
  bool constant_padding_;
  friend struct ::TableStruct_mediapipe_2fcalculators_2fimage_2fimage_5ftransformation_5fcalculator_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// ImageTransformationCalculatorOptions

// optional int32 output_width = 1 [default = 0];
inline bool ImageTransformationCalculatorOptions::_internal_has_output_width() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool ImageTransformationCalculatorOptions::has_output_width() const {
  return _internal_has_output_width();
}
inline void ImageTransformationCalculatorOptions::clear_output_width() {
  output_width_ = 0;
  _has_bits_[0] &= ~0x00000001u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 ImageTransformationCalculatorOptions::_internal_output_width() const {
  return output_width_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 ImageTransformationCalculatorOptions::output_width() const {
  // @@protoc_insertion_point(field_get:mediapipe.ImageTransformationCalculatorOptions.output_width)
  return _internal_output_width();
}
inline void ImageTransformationCalculatorOptions::_internal_set_output_width(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000001u;
  output_width_ = value;
}
inline void ImageTransformationCalculatorOptions::set_output_width(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_output_width(value);
  // @@protoc_insertion_point(field_set:mediapipe.ImageTransformationCalculatorOptions.output_width)
}

// optional int32 output_height = 2 [default = 0];
inline bool ImageTransformationCalculatorOptions::_internal_has_output_height() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool ImageTransformationCalculatorOptions::has_output_height() const {
  return _internal_has_output_height();
}
inline void ImageTransformationCalculatorOptions::clear_output_height() {
  output_height_ = 0;
  _has_bits_[0] &= ~0x00000002u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 ImageTransformationCalculatorOptions::_internal_output_height() const {
  return output_height_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 ImageTransformationCalculatorOptions::output_height() const {
  // @@protoc_insertion_point(field_get:mediapipe.ImageTransformationCalculatorOptions.output_height)
  return _internal_output_height();
}
inline void ImageTransformationCalculatorOptions::_internal_set_output_height(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000002u;
  output_height_ = value;
}
inline void ImageTransformationCalculatorOptions::set_output_height(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_output_height(value);
  // @@protoc_insertion_point(field_set:mediapipe.ImageTransformationCalculatorOptions.output_height)
}

// optional .mediapipe.RotationMode.Mode rotation_mode = 3;
inline bool ImageTransformationCalculatorOptions::_internal_has_rotation_mode() const {
  bool value = (_has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool ImageTransformationCalculatorOptions::has_rotation_mode() const {
  return _internal_has_rotation_mode();
}
inline void ImageTransformationCalculatorOptions::clear_rotation_mode() {
  rotation_mode_ = 0;
  _has_bits_[0] &= ~0x00000004u;
}
inline ::mediapipe::RotationMode_Mode ImageTransformationCalculatorOptions::_internal_rotation_mode() const {
  return static_cast< ::mediapipe::RotationMode_Mode >(rotation_mode_);
}
inline ::mediapipe::RotationMode_Mode ImageTransformationCalculatorOptions::rotation_mode() const {
  // @@protoc_insertion_point(field_get:mediapipe.ImageTransformationCalculatorOptions.rotation_mode)
  return _internal_rotation_mode();
}
inline void ImageTransformationCalculatorOptions::_internal_set_rotation_mode(::mediapipe::RotationMode_Mode value) {
  assert(::mediapipe::RotationMode_Mode_IsValid(value));
  _has_bits_[0] |= 0x00000004u;
  rotation_mode_ = value;
}
inline void ImageTransformationCalculatorOptions::set_rotation_mode(::mediapipe::RotationMode_Mode value) {
  _internal_set_rotation_mode(value);
  // @@protoc_insertion_point(field_set:mediapipe.ImageTransformationCalculatorOptions.rotation_mode)
}

// optional bool flip_vertically = 4 [default = false];
inline bool ImageTransformationCalculatorOptions::_internal_has_flip_vertically() const {
  bool value = (_has_bits_[0] & 0x00000008u) != 0;
  return value;
}
inline bool ImageTransformationCalculatorOptions::has_flip_vertically() const {
  return _internal_has_flip_vertically();
}
inline void ImageTransformationCalculatorOptions::clear_flip_vertically() {
  flip_vertically_ = false;
  _has_bits_[0] &= ~0x00000008u;
}
inline bool ImageTransformationCalculatorOptions::_internal_flip_vertically() const {
  return flip_vertically_;
}
inline bool ImageTransformationCalculatorOptions::flip_vertically() const {
  // @@protoc_insertion_point(field_get:mediapipe.ImageTransformationCalculatorOptions.flip_vertically)
  return _internal_flip_vertically();
}
inline void ImageTransformationCalculatorOptions::_internal_set_flip_vertically(bool value) {
  _has_bits_[0] |= 0x00000008u;
  flip_vertically_ = value;
}
inline void ImageTransformationCalculatorOptions::set_flip_vertically(bool value) {
  _internal_set_flip_vertically(value);
  // @@protoc_insertion_point(field_set:mediapipe.ImageTransformationCalculatorOptions.flip_vertically)
}

// optional bool flip_horizontally = 5 [default = false];
inline bool ImageTransformationCalculatorOptions::_internal_has_flip_horizontally() const {
  bool value = (_has_bits_[0] & 0x00000010u) != 0;
  return value;
}
inline bool ImageTransformationCalculatorOptions::has_flip_horizontally() const {
  return _internal_has_flip_horizontally();
}
inline void ImageTransformationCalculatorOptions::clear_flip_horizontally() {
  flip_horizontally_ = false;
  _has_bits_[0] &= ~0x00000010u;
}
inline bool ImageTransformationCalculatorOptions::_internal_flip_horizontally() const {
  return flip_horizontally_;
}
inline bool ImageTransformationCalculatorOptions::flip_horizontally() const {
  // @@protoc_insertion_point(field_get:mediapipe.ImageTransformationCalculatorOptions.flip_horizontally)
  return _internal_flip_horizontally();
}
inline void ImageTransformationCalculatorOptions::_internal_set_flip_horizontally(bool value) {
  _has_bits_[0] |= 0x00000010u;
  flip_horizontally_ = value;
}
inline void ImageTransformationCalculatorOptions::set_flip_horizontally(bool value) {
  _internal_set_flip_horizontally(value);
  // @@protoc_insertion_point(field_set:mediapipe.ImageTransformationCalculatorOptions.flip_horizontally)
}

// optional .mediapipe.ScaleMode.Mode scale_mode = 6;
inline bool ImageTransformationCalculatorOptions::_internal_has_scale_mode() const {
  bool value = (_has_bits_[0] & 0x00000020u) != 0;
  return value;
}
inline bool ImageTransformationCalculatorOptions::has_scale_mode() const {
  return _internal_has_scale_mode();
}
inline void ImageTransformationCalculatorOptions::clear_scale_mode() {
  scale_mode_ = 0;
  _has_bits_[0] &= ~0x00000020u;
}
inline ::mediapipe::ScaleMode_Mode ImageTransformationCalculatorOptions::_internal_scale_mode() const {
  return static_cast< ::mediapipe::ScaleMode_Mode >(scale_mode_);
}
inline ::mediapipe::ScaleMode_Mode ImageTransformationCalculatorOptions::scale_mode() const {
  // @@protoc_insertion_point(field_get:mediapipe.ImageTransformationCalculatorOptions.scale_mode)
  return _internal_scale_mode();
}
inline void ImageTransformationCalculatorOptions::_internal_set_scale_mode(::mediapipe::ScaleMode_Mode value) {
  assert(::mediapipe::ScaleMode_Mode_IsValid(value));
  _has_bits_[0] |= 0x00000020u;
  scale_mode_ = value;
}
inline void ImageTransformationCalculatorOptions::set_scale_mode(::mediapipe::ScaleMode_Mode value) {
  _internal_set_scale_mode(value);
  // @@protoc_insertion_point(field_set:mediapipe.ImageTransformationCalculatorOptions.scale_mode)
}

// optional bool constant_padding = 7 [default = true];
inline bool ImageTransformationCalculatorOptions::_internal_has_constant_padding() const {
  bool value = (_has_bits_[0] & 0x00000040u) != 0;
  return value;
}
inline bool ImageTransformationCalculatorOptions::has_constant_padding() const {
  return _internal_has_constant_padding();
}
inline void ImageTransformationCalculatorOptions::clear_constant_padding() {
  constant_padding_ = true;
  _has_bits_[0] &= ~0x00000040u;
}
inline bool ImageTransformationCalculatorOptions::_internal_constant_padding() const {
  return constant_padding_;
}
inline bool ImageTransformationCalculatorOptions::constant_padding() const {
  // @@protoc_insertion_point(field_get:mediapipe.ImageTransformationCalculatorOptions.constant_padding)
  return _internal_constant_padding();
}
inline void ImageTransformationCalculatorOptions::_internal_set_constant_padding(bool value) {
  _has_bits_[0] |= 0x00000040u;
  constant_padding_ = value;
}
inline void ImageTransformationCalculatorOptions::set_constant_padding(bool value) {
  _internal_set_constant_padding(value);
  // @@protoc_insertion_point(field_set:mediapipe.ImageTransformationCalculatorOptions.constant_padding)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <x/google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fimage_2fimage_5ftransformation_5fcalculator_2eproto
