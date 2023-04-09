// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/tflite/tflite_tensors_to_segmentation_calculator.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fsegmentation_5fcalculator_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fsegmentation_5fcalculator_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fsegmentation_5fcalculator_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fsegmentation_5fcalculator_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fsegmentation_5fcalculator_2eproto;
namespace mediapipe {
class TfLiteTensorsToSegmentationCalculatorOptions;
struct TfLiteTensorsToSegmentationCalculatorOptionsDefaultTypeInternal;
extern TfLiteTensorsToSegmentationCalculatorOptionsDefaultTypeInternal _TfLiteTensorsToSegmentationCalculatorOptions_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::TfLiteTensorsToSegmentationCalculatorOptions* Arena::CreateMaybeMessage<::mediapipe::TfLiteTensorsToSegmentationCalculatorOptions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class TfLiteTensorsToSegmentationCalculatorOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.TfLiteTensorsToSegmentationCalculatorOptions) */ {
 public:
  inline TfLiteTensorsToSegmentationCalculatorOptions() : TfLiteTensorsToSegmentationCalculatorOptions(nullptr) {}
  ~TfLiteTensorsToSegmentationCalculatorOptions() override;
  explicit constexpr TfLiteTensorsToSegmentationCalculatorOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  TfLiteTensorsToSegmentationCalculatorOptions(const TfLiteTensorsToSegmentationCalculatorOptions& from);
  TfLiteTensorsToSegmentationCalculatorOptions(TfLiteTensorsToSegmentationCalculatorOptions&& from) noexcept
    : TfLiteTensorsToSegmentationCalculatorOptions() {
    *this = ::std::move(from);
  }

  inline TfLiteTensorsToSegmentationCalculatorOptions& operator=(const TfLiteTensorsToSegmentationCalculatorOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline TfLiteTensorsToSegmentationCalculatorOptions& operator=(TfLiteTensorsToSegmentationCalculatorOptions&& from) noexcept {
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
  static const TfLiteTensorsToSegmentationCalculatorOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const TfLiteTensorsToSegmentationCalculatorOptions* internal_default_instance() {
    return reinterpret_cast<const TfLiteTensorsToSegmentationCalculatorOptions*>(
               &_TfLiteTensorsToSegmentationCalculatorOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(TfLiteTensorsToSegmentationCalculatorOptions& a, TfLiteTensorsToSegmentationCalculatorOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(TfLiteTensorsToSegmentationCalculatorOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(TfLiteTensorsToSegmentationCalculatorOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline TfLiteTensorsToSegmentationCalculatorOptions* New() const final {
    return CreateMaybeMessage<TfLiteTensorsToSegmentationCalculatorOptions>(nullptr);
  }

  TfLiteTensorsToSegmentationCalculatorOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<TfLiteTensorsToSegmentationCalculatorOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const TfLiteTensorsToSegmentationCalculatorOptions& from);
  void MergeFrom(const TfLiteTensorsToSegmentationCalculatorOptions& from);
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
  void InternalSwap(TfLiteTensorsToSegmentationCalculatorOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.TfLiteTensorsToSegmentationCalculatorOptions";
  }
  protected:
  explicit TfLiteTensorsToSegmentationCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kTensorWidthFieldNumber = 1,
    kTensorHeightFieldNumber = 2,
    kTensorChannelsFieldNumber = 3,
    kFlipVerticallyFieldNumber = 6,
    kCombineWithPreviousRatioFieldNumber = 4,
    kOutputLayerIndexFieldNumber = 5,
  };
  // optional int32 tensor_width = 1;
  bool has_tensor_width() const;
  private:
  bool _internal_has_tensor_width() const;
  public:
  void clear_tensor_width();
  ::PROTOBUF_NAMESPACE_ID::int32 tensor_width() const;
  void set_tensor_width(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_tensor_width() const;
  void _internal_set_tensor_width(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional int32 tensor_height = 2;
  bool has_tensor_height() const;
  private:
  bool _internal_has_tensor_height() const;
  public:
  void clear_tensor_height();
  ::PROTOBUF_NAMESPACE_ID::int32 tensor_height() const;
  void set_tensor_height(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_tensor_height() const;
  void _internal_set_tensor_height(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional int32 tensor_channels = 3;
  bool has_tensor_channels() const;
  private:
  bool _internal_has_tensor_channels() const;
  public:
  void clear_tensor_channels();
  ::PROTOBUF_NAMESPACE_ID::int32 tensor_channels() const;
  void set_tensor_channels(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_tensor_channels() const;
  void _internal_set_tensor_channels(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional bool flip_vertically = 6;
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

  // optional float combine_with_previous_ratio = 4 [default = 1];
  bool has_combine_with_previous_ratio() const;
  private:
  bool _internal_has_combine_with_previous_ratio() const;
  public:
  void clear_combine_with_previous_ratio();
  float combine_with_previous_ratio() const;
  void set_combine_with_previous_ratio(float value);
  private:
  float _internal_combine_with_previous_ratio() const;
  void _internal_set_combine_with_previous_ratio(float value);
  public:

  // optional int32 output_layer_index = 5 [default = 1];
  bool has_output_layer_index() const;
  private:
  bool _internal_has_output_layer_index() const;
  public:
  void clear_output_layer_index();
  ::PROTOBUF_NAMESPACE_ID::int32 output_layer_index() const;
  void set_output_layer_index(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_output_layer_index() const;
  void _internal_set_output_layer_index(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  static const int kExtFieldNumber = 252526026;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::TfLiteTensorsToSegmentationCalculatorOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.TfLiteTensorsToSegmentationCalculatorOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::int32 tensor_width_;
  ::PROTOBUF_NAMESPACE_ID::int32 tensor_height_;
  ::PROTOBUF_NAMESPACE_ID::int32 tensor_channels_;
  bool flip_vertically_;
  float combine_with_previous_ratio_;
  ::PROTOBUF_NAMESPACE_ID::int32 output_layer_index_;
  friend struct ::TableStruct_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fsegmentation_5fcalculator_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// TfLiteTensorsToSegmentationCalculatorOptions

// optional int32 tensor_width = 1;
inline bool TfLiteTensorsToSegmentationCalculatorOptions::_internal_has_tensor_width() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool TfLiteTensorsToSegmentationCalculatorOptions::has_tensor_width() const {
  return _internal_has_tensor_width();
}
inline void TfLiteTensorsToSegmentationCalculatorOptions::clear_tensor_width() {
  tensor_width_ = 0;
  _has_bits_[0] &= ~0x00000001u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 TfLiteTensorsToSegmentationCalculatorOptions::_internal_tensor_width() const {
  return tensor_width_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 TfLiteTensorsToSegmentationCalculatorOptions::tensor_width() const {
  // @@protoc_insertion_point(field_get:mediapipe.TfLiteTensorsToSegmentationCalculatorOptions.tensor_width)
  return _internal_tensor_width();
}
inline void TfLiteTensorsToSegmentationCalculatorOptions::_internal_set_tensor_width(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000001u;
  tensor_width_ = value;
}
inline void TfLiteTensorsToSegmentationCalculatorOptions::set_tensor_width(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_tensor_width(value);
  // @@protoc_insertion_point(field_set:mediapipe.TfLiteTensorsToSegmentationCalculatorOptions.tensor_width)
}

// optional int32 tensor_height = 2;
inline bool TfLiteTensorsToSegmentationCalculatorOptions::_internal_has_tensor_height() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool TfLiteTensorsToSegmentationCalculatorOptions::has_tensor_height() const {
  return _internal_has_tensor_height();
}
inline void TfLiteTensorsToSegmentationCalculatorOptions::clear_tensor_height() {
  tensor_height_ = 0;
  _has_bits_[0] &= ~0x00000002u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 TfLiteTensorsToSegmentationCalculatorOptions::_internal_tensor_height() const {
  return tensor_height_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 TfLiteTensorsToSegmentationCalculatorOptions::tensor_height() const {
  // @@protoc_insertion_point(field_get:mediapipe.TfLiteTensorsToSegmentationCalculatorOptions.tensor_height)
  return _internal_tensor_height();
}
inline void TfLiteTensorsToSegmentationCalculatorOptions::_internal_set_tensor_height(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000002u;
  tensor_height_ = value;
}
inline void TfLiteTensorsToSegmentationCalculatorOptions::set_tensor_height(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_tensor_height(value);
  // @@protoc_insertion_point(field_set:mediapipe.TfLiteTensorsToSegmentationCalculatorOptions.tensor_height)
}

// optional int32 tensor_channels = 3;
inline bool TfLiteTensorsToSegmentationCalculatorOptions::_internal_has_tensor_channels() const {
  bool value = (_has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool TfLiteTensorsToSegmentationCalculatorOptions::has_tensor_channels() const {
  return _internal_has_tensor_channels();
}
inline void TfLiteTensorsToSegmentationCalculatorOptions::clear_tensor_channels() {
  tensor_channels_ = 0;
  _has_bits_[0] &= ~0x00000004u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 TfLiteTensorsToSegmentationCalculatorOptions::_internal_tensor_channels() const {
  return tensor_channels_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 TfLiteTensorsToSegmentationCalculatorOptions::tensor_channels() const {
  // @@protoc_insertion_point(field_get:mediapipe.TfLiteTensorsToSegmentationCalculatorOptions.tensor_channels)
  return _internal_tensor_channels();
}
inline void TfLiteTensorsToSegmentationCalculatorOptions::_internal_set_tensor_channels(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000004u;
  tensor_channels_ = value;
}
inline void TfLiteTensorsToSegmentationCalculatorOptions::set_tensor_channels(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_tensor_channels(value);
  // @@protoc_insertion_point(field_set:mediapipe.TfLiteTensorsToSegmentationCalculatorOptions.tensor_channels)
}

// optional float combine_with_previous_ratio = 4 [default = 1];
inline bool TfLiteTensorsToSegmentationCalculatorOptions::_internal_has_combine_with_previous_ratio() const {
  bool value = (_has_bits_[0] & 0x00000010u) != 0;
  return value;
}
inline bool TfLiteTensorsToSegmentationCalculatorOptions::has_combine_with_previous_ratio() const {
  return _internal_has_combine_with_previous_ratio();
}
inline void TfLiteTensorsToSegmentationCalculatorOptions::clear_combine_with_previous_ratio() {
  combine_with_previous_ratio_ = 1;
  _has_bits_[0] &= ~0x00000010u;
}
inline float TfLiteTensorsToSegmentationCalculatorOptions::_internal_combine_with_previous_ratio() const {
  return combine_with_previous_ratio_;
}
inline float TfLiteTensorsToSegmentationCalculatorOptions::combine_with_previous_ratio() const {
  // @@protoc_insertion_point(field_get:mediapipe.TfLiteTensorsToSegmentationCalculatorOptions.combine_with_previous_ratio)
  return _internal_combine_with_previous_ratio();
}
inline void TfLiteTensorsToSegmentationCalculatorOptions::_internal_set_combine_with_previous_ratio(float value) {
  _has_bits_[0] |= 0x00000010u;
  combine_with_previous_ratio_ = value;
}
inline void TfLiteTensorsToSegmentationCalculatorOptions::set_combine_with_previous_ratio(float value) {
  _internal_set_combine_with_previous_ratio(value);
  // @@protoc_insertion_point(field_set:mediapipe.TfLiteTensorsToSegmentationCalculatorOptions.combine_with_previous_ratio)
}

// optional int32 output_layer_index = 5 [default = 1];
inline bool TfLiteTensorsToSegmentationCalculatorOptions::_internal_has_output_layer_index() const {
  bool value = (_has_bits_[0] & 0x00000020u) != 0;
  return value;
}
inline bool TfLiteTensorsToSegmentationCalculatorOptions::has_output_layer_index() const {
  return _internal_has_output_layer_index();
}
inline void TfLiteTensorsToSegmentationCalculatorOptions::clear_output_layer_index() {
  output_layer_index_ = 1;
  _has_bits_[0] &= ~0x00000020u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 TfLiteTensorsToSegmentationCalculatorOptions::_internal_output_layer_index() const {
  return output_layer_index_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 TfLiteTensorsToSegmentationCalculatorOptions::output_layer_index() const {
  // @@protoc_insertion_point(field_get:mediapipe.TfLiteTensorsToSegmentationCalculatorOptions.output_layer_index)
  return _internal_output_layer_index();
}
inline void TfLiteTensorsToSegmentationCalculatorOptions::_internal_set_output_layer_index(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000020u;
  output_layer_index_ = value;
}
inline void TfLiteTensorsToSegmentationCalculatorOptions::set_output_layer_index(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_output_layer_index(value);
  // @@protoc_insertion_point(field_set:mediapipe.TfLiteTensorsToSegmentationCalculatorOptions.output_layer_index)
}

// optional bool flip_vertically = 6;
inline bool TfLiteTensorsToSegmentationCalculatorOptions::_internal_has_flip_vertically() const {
  bool value = (_has_bits_[0] & 0x00000008u) != 0;
  return value;
}
inline bool TfLiteTensorsToSegmentationCalculatorOptions::has_flip_vertically() const {
  return _internal_has_flip_vertically();
}
inline void TfLiteTensorsToSegmentationCalculatorOptions::clear_flip_vertically() {
  flip_vertically_ = false;
  _has_bits_[0] &= ~0x00000008u;
}
inline bool TfLiteTensorsToSegmentationCalculatorOptions::_internal_flip_vertically() const {
  return flip_vertically_;
}
inline bool TfLiteTensorsToSegmentationCalculatorOptions::flip_vertically() const {
  // @@protoc_insertion_point(field_get:mediapipe.TfLiteTensorsToSegmentationCalculatorOptions.flip_vertically)
  return _internal_flip_vertically();
}
inline void TfLiteTensorsToSegmentationCalculatorOptions::_internal_set_flip_vertically(bool value) {
  _has_bits_[0] |= 0x00000008u;
  flip_vertically_ = value;
}
inline void TfLiteTensorsToSegmentationCalculatorOptions::set_flip_vertically(bool value) {
  _internal_set_flip_vertically(value);
  // @@protoc_insertion_point(field_set:mediapipe.TfLiteTensorsToSegmentationCalculatorOptions.flip_vertically)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <x/google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fsegmentation_5fcalculator_2eproto
