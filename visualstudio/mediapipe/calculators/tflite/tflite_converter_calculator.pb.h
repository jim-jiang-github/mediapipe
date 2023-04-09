// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/tflite/tflite_converter_calculator.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto;
namespace mediapipe {
class TfLiteConverterCalculatorOptions;
struct TfLiteConverterCalculatorOptionsDefaultTypeInternal;
extern TfLiteConverterCalculatorOptionsDefaultTypeInternal _TfLiteConverterCalculatorOptions_default_instance_;
class TfLiteConverterCalculatorOptions_TensorFloatRange;
struct TfLiteConverterCalculatorOptions_TensorFloatRangeDefaultTypeInternal;
extern TfLiteConverterCalculatorOptions_TensorFloatRangeDefaultTypeInternal _TfLiteConverterCalculatorOptions_TensorFloatRange_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::TfLiteConverterCalculatorOptions* Arena::CreateMaybeMessage<::mediapipe::TfLiteConverterCalculatorOptions>(Arena*);
template<> ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange* Arena::CreateMaybeMessage<::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class TfLiteConverterCalculatorOptions_TensorFloatRange PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange) */ {
 public:
  inline TfLiteConverterCalculatorOptions_TensorFloatRange() : TfLiteConverterCalculatorOptions_TensorFloatRange(nullptr) {}
  ~TfLiteConverterCalculatorOptions_TensorFloatRange() override;
  explicit constexpr TfLiteConverterCalculatorOptions_TensorFloatRange(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  TfLiteConverterCalculatorOptions_TensorFloatRange(const TfLiteConverterCalculatorOptions_TensorFloatRange& from);
  TfLiteConverterCalculatorOptions_TensorFloatRange(TfLiteConverterCalculatorOptions_TensorFloatRange&& from) noexcept
    : TfLiteConverterCalculatorOptions_TensorFloatRange() {
    *this = ::std::move(from);
  }

  inline TfLiteConverterCalculatorOptions_TensorFloatRange& operator=(const TfLiteConverterCalculatorOptions_TensorFloatRange& from) {
    CopyFrom(from);
    return *this;
  }
  inline TfLiteConverterCalculatorOptions_TensorFloatRange& operator=(TfLiteConverterCalculatorOptions_TensorFloatRange&& from) noexcept {
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
  static const TfLiteConverterCalculatorOptions_TensorFloatRange& default_instance() {
    return *internal_default_instance();
  }
  static inline const TfLiteConverterCalculatorOptions_TensorFloatRange* internal_default_instance() {
    return reinterpret_cast<const TfLiteConverterCalculatorOptions_TensorFloatRange*>(
               &_TfLiteConverterCalculatorOptions_TensorFloatRange_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(TfLiteConverterCalculatorOptions_TensorFloatRange& a, TfLiteConverterCalculatorOptions_TensorFloatRange& b) {
    a.Swap(&b);
  }
  inline void Swap(TfLiteConverterCalculatorOptions_TensorFloatRange* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(TfLiteConverterCalculatorOptions_TensorFloatRange* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline TfLiteConverterCalculatorOptions_TensorFloatRange* New() const final {
    return CreateMaybeMessage<TfLiteConverterCalculatorOptions_TensorFloatRange>(nullptr);
  }

  TfLiteConverterCalculatorOptions_TensorFloatRange* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<TfLiteConverterCalculatorOptions_TensorFloatRange>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const TfLiteConverterCalculatorOptions_TensorFloatRange& from);
  void MergeFrom(const TfLiteConverterCalculatorOptions_TensorFloatRange& from);
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
  void InternalSwap(TfLiteConverterCalculatorOptions_TensorFloatRange* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange";
  }
  protected:
  explicit TfLiteConverterCalculatorOptions_TensorFloatRange(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kMinFieldNumber = 1,
    kMaxFieldNumber = 2,
  };
  // optional float min = 1;
  bool has_min() const;
  private:
  bool _internal_has_min() const;
  public:
  void clear_min();
  float min() const;
  void set_min(float value);
  private:
  float _internal_min() const;
  void _internal_set_min(float value);
  public:

  // optional float max = 2;
  bool has_max() const;
  private:
  bool _internal_has_max() const;
  public:
  void clear_max();
  float max() const;
  void set_max(float value);
  private:
  float _internal_max() const;
  void _internal_set_max(float value);
  public:

  // @@protoc_insertion_point(class_scope:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  float min_;
  float max_;
  friend struct ::TableStruct_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto;
};
// -------------------------------------------------------------------

class TfLiteConverterCalculatorOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.TfLiteConverterCalculatorOptions) */ {
 public:
  inline TfLiteConverterCalculatorOptions() : TfLiteConverterCalculatorOptions(nullptr) {}
  ~TfLiteConverterCalculatorOptions() override;
  explicit constexpr TfLiteConverterCalculatorOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  TfLiteConverterCalculatorOptions(const TfLiteConverterCalculatorOptions& from);
  TfLiteConverterCalculatorOptions(TfLiteConverterCalculatorOptions&& from) noexcept
    : TfLiteConverterCalculatorOptions() {
    *this = ::std::move(from);
  }

  inline TfLiteConverterCalculatorOptions& operator=(const TfLiteConverterCalculatorOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline TfLiteConverterCalculatorOptions& operator=(TfLiteConverterCalculatorOptions&& from) noexcept {
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
  static const TfLiteConverterCalculatorOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const TfLiteConverterCalculatorOptions* internal_default_instance() {
    return reinterpret_cast<const TfLiteConverterCalculatorOptions*>(
               &_TfLiteConverterCalculatorOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(TfLiteConverterCalculatorOptions& a, TfLiteConverterCalculatorOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(TfLiteConverterCalculatorOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(TfLiteConverterCalculatorOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline TfLiteConverterCalculatorOptions* New() const final {
    return CreateMaybeMessage<TfLiteConverterCalculatorOptions>(nullptr);
  }

  TfLiteConverterCalculatorOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<TfLiteConverterCalculatorOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const TfLiteConverterCalculatorOptions& from);
  void MergeFrom(const TfLiteConverterCalculatorOptions& from);
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
  void InternalSwap(TfLiteConverterCalculatorOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.TfLiteConverterCalculatorOptions";
  }
  protected:
  explicit TfLiteConverterCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef TfLiteConverterCalculatorOptions_TensorFloatRange TensorFloatRange;

  // accessors -------------------------------------------------------

  enum : int {
    kOutputTensorFloatRangeFieldNumber = 9,
    kUseCustomNormalizationFieldNumber = 6,
    kFlipVerticallyFieldNumber = 2,
    kRowMajorMatrixFieldNumber = 4,
    kUseQuantizedTensorsFieldNumber = 5,
    kZeroCenterFieldNumber = 1,
    kMaxNumChannelsFieldNumber = 3,
    kCustomDivFieldNumber = 7,
    kCustomSubFieldNumber = 8,
  };
  // optional .mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange output_tensor_float_range = 9;
  bool has_output_tensor_float_range() const;
  private:
  bool _internal_has_output_tensor_float_range() const;
  public:
  void clear_output_tensor_float_range();
  const ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange& output_tensor_float_range() const;
  PROTOBUF_FUTURE_MUST_USE_RESULT ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange* release_output_tensor_float_range();
  ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange* mutable_output_tensor_float_range();
  void set_allocated_output_tensor_float_range(::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange* output_tensor_float_range);
  private:
  const ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange& _internal_output_tensor_float_range() const;
  ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange* _internal_mutable_output_tensor_float_range();
  public:
  void unsafe_arena_set_allocated_output_tensor_float_range(
      ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange* output_tensor_float_range);
  ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange* unsafe_arena_release_output_tensor_float_range();

  // optional bool use_custom_normalization = 6 [default = false];
  bool has_use_custom_normalization() const;
  private:
  bool _internal_has_use_custom_normalization() const;
  public:
  void clear_use_custom_normalization();
  bool use_custom_normalization() const;
  void set_use_custom_normalization(bool value);
  private:
  bool _internal_use_custom_normalization() const;
  void _internal_set_use_custom_normalization(bool value);
  public:

  // optional bool flip_vertically = 2 [default = false];
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

  // optional bool row_major_matrix = 4 [default = false];
  bool has_row_major_matrix() const;
  private:
  bool _internal_has_row_major_matrix() const;
  public:
  void clear_row_major_matrix();
  bool row_major_matrix() const;
  void set_row_major_matrix(bool value);
  private:
  bool _internal_row_major_matrix() const;
  void _internal_set_row_major_matrix(bool value);
  public:

  // optional bool use_quantized_tensors = 5 [default = false];
  bool has_use_quantized_tensors() const;
  private:
  bool _internal_has_use_quantized_tensors() const;
  public:
  void clear_use_quantized_tensors();
  bool use_quantized_tensors() const;
  void set_use_quantized_tensors(bool value);
  private:
  bool _internal_use_quantized_tensors() const;
  void _internal_set_use_quantized_tensors(bool value);
  public:

  // optional bool zero_center = 1 [default = true];
  bool has_zero_center() const;
  private:
  bool _internal_has_zero_center() const;
  public:
  void clear_zero_center();
  bool zero_center() const;
  void set_zero_center(bool value);
  private:
  bool _internal_zero_center() const;
  void _internal_set_zero_center(bool value);
  public:

  // optional int32 max_num_channels = 3 [default = 3];
  bool has_max_num_channels() const;
  private:
  bool _internal_has_max_num_channels() const;
  public:
  void clear_max_num_channels();
  ::PROTOBUF_NAMESPACE_ID::int32 max_num_channels() const;
  void set_max_num_channels(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_max_num_channels() const;
  void _internal_set_max_num_channels(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional float custom_div = 7 [default = -1];
  bool has_custom_div() const;
  private:
  bool _internal_has_custom_div() const;
  public:
  void clear_custom_div();
  float custom_div() const;
  void set_custom_div(float value);
  private:
  float _internal_custom_div() const;
  void _internal_set_custom_div(float value);
  public:

  // optional float custom_sub = 8 [default = -1];
  bool has_custom_sub() const;
  private:
  bool _internal_has_custom_sub() const;
  public:
  void clear_custom_sub();
  float custom_sub() const;
  void set_custom_sub(float value);
  private:
  float _internal_custom_sub() const;
  void _internal_set_custom_sub(float value);
  public:

  static const int kExtFieldNumber = 245817797;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::TfLiteConverterCalculatorOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.TfLiteConverterCalculatorOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange* output_tensor_float_range_;
  bool use_custom_normalization_;
  bool flip_vertically_;
  bool row_major_matrix_;
  bool use_quantized_tensors_;
  bool zero_center_;
  ::PROTOBUF_NAMESPACE_ID::int32 max_num_channels_;
  float custom_div_;
  float custom_sub_;
  friend struct ::TableStruct_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// TfLiteConverterCalculatorOptions_TensorFloatRange

// optional float min = 1;
inline bool TfLiteConverterCalculatorOptions_TensorFloatRange::_internal_has_min() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool TfLiteConverterCalculatorOptions_TensorFloatRange::has_min() const {
  return _internal_has_min();
}
inline void TfLiteConverterCalculatorOptions_TensorFloatRange::clear_min() {
  min_ = 0;
  _has_bits_[0] &= ~0x00000001u;
}
inline float TfLiteConverterCalculatorOptions_TensorFloatRange::_internal_min() const {
  return min_;
}
inline float TfLiteConverterCalculatorOptions_TensorFloatRange::min() const {
  // @@protoc_insertion_point(field_get:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange.min)
  return _internal_min();
}
inline void TfLiteConverterCalculatorOptions_TensorFloatRange::_internal_set_min(float value) {
  _has_bits_[0] |= 0x00000001u;
  min_ = value;
}
inline void TfLiteConverterCalculatorOptions_TensorFloatRange::set_min(float value) {
  _internal_set_min(value);
  // @@protoc_insertion_point(field_set:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange.min)
}

// optional float max = 2;
inline bool TfLiteConverterCalculatorOptions_TensorFloatRange::_internal_has_max() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool TfLiteConverterCalculatorOptions_TensorFloatRange::has_max() const {
  return _internal_has_max();
}
inline void TfLiteConverterCalculatorOptions_TensorFloatRange::clear_max() {
  max_ = 0;
  _has_bits_[0] &= ~0x00000002u;
}
inline float TfLiteConverterCalculatorOptions_TensorFloatRange::_internal_max() const {
  return max_;
}
inline float TfLiteConverterCalculatorOptions_TensorFloatRange::max() const {
  // @@protoc_insertion_point(field_get:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange.max)
  return _internal_max();
}
inline void TfLiteConverterCalculatorOptions_TensorFloatRange::_internal_set_max(float value) {
  _has_bits_[0] |= 0x00000002u;
  max_ = value;
}
inline void TfLiteConverterCalculatorOptions_TensorFloatRange::set_max(float value) {
  _internal_set_max(value);
  // @@protoc_insertion_point(field_set:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange.max)
}

// -------------------------------------------------------------------

// TfLiteConverterCalculatorOptions

// optional bool zero_center = 1 [default = true];
inline bool TfLiteConverterCalculatorOptions::_internal_has_zero_center() const {
  bool value = (_has_bits_[0] & 0x00000020u) != 0;
  return value;
}
inline bool TfLiteConverterCalculatorOptions::has_zero_center() const {
  return _internal_has_zero_center();
}
inline void TfLiteConverterCalculatorOptions::clear_zero_center() {
  zero_center_ = true;
  _has_bits_[0] &= ~0x00000020u;
}
inline bool TfLiteConverterCalculatorOptions::_internal_zero_center() const {
  return zero_center_;
}
inline bool TfLiteConverterCalculatorOptions::zero_center() const {
  // @@protoc_insertion_point(field_get:mediapipe.TfLiteConverterCalculatorOptions.zero_center)
  return _internal_zero_center();
}
inline void TfLiteConverterCalculatorOptions::_internal_set_zero_center(bool value) {
  _has_bits_[0] |= 0x00000020u;
  zero_center_ = value;
}
inline void TfLiteConverterCalculatorOptions::set_zero_center(bool value) {
  _internal_set_zero_center(value);
  // @@protoc_insertion_point(field_set:mediapipe.TfLiteConverterCalculatorOptions.zero_center)
}

// optional bool use_custom_normalization = 6 [default = false];
inline bool TfLiteConverterCalculatorOptions::_internal_has_use_custom_normalization() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool TfLiteConverterCalculatorOptions::has_use_custom_normalization() const {
  return _internal_has_use_custom_normalization();
}
inline void TfLiteConverterCalculatorOptions::clear_use_custom_normalization() {
  use_custom_normalization_ = false;
  _has_bits_[0] &= ~0x00000002u;
}
inline bool TfLiteConverterCalculatorOptions::_internal_use_custom_normalization() const {
  return use_custom_normalization_;
}
inline bool TfLiteConverterCalculatorOptions::use_custom_normalization() const {
  // @@protoc_insertion_point(field_get:mediapipe.TfLiteConverterCalculatorOptions.use_custom_normalization)
  return _internal_use_custom_normalization();
}
inline void TfLiteConverterCalculatorOptions::_internal_set_use_custom_normalization(bool value) {
  _has_bits_[0] |= 0x00000002u;
  use_custom_normalization_ = value;
}
inline void TfLiteConverterCalculatorOptions::set_use_custom_normalization(bool value) {
  _internal_set_use_custom_normalization(value);
  // @@protoc_insertion_point(field_set:mediapipe.TfLiteConverterCalculatorOptions.use_custom_normalization)
}

// optional float custom_div = 7 [default = -1];
inline bool TfLiteConverterCalculatorOptions::_internal_has_custom_div() const {
  bool value = (_has_bits_[0] & 0x00000080u) != 0;
  return value;
}
inline bool TfLiteConverterCalculatorOptions::has_custom_div() const {
  return _internal_has_custom_div();
}
inline void TfLiteConverterCalculatorOptions::clear_custom_div() {
  custom_div_ = -1;
  _has_bits_[0] &= ~0x00000080u;
}
inline float TfLiteConverterCalculatorOptions::_internal_custom_div() const {
  return custom_div_;
}
inline float TfLiteConverterCalculatorOptions::custom_div() const {
  // @@protoc_insertion_point(field_get:mediapipe.TfLiteConverterCalculatorOptions.custom_div)
  return _internal_custom_div();
}
inline void TfLiteConverterCalculatorOptions::_internal_set_custom_div(float value) {
  _has_bits_[0] |= 0x00000080u;
  custom_div_ = value;
}
inline void TfLiteConverterCalculatorOptions::set_custom_div(float value) {
  _internal_set_custom_div(value);
  // @@protoc_insertion_point(field_set:mediapipe.TfLiteConverterCalculatorOptions.custom_div)
}

// optional float custom_sub = 8 [default = -1];
inline bool TfLiteConverterCalculatorOptions::_internal_has_custom_sub() const {
  bool value = (_has_bits_[0] & 0x00000100u) != 0;
  return value;
}
inline bool TfLiteConverterCalculatorOptions::has_custom_sub() const {
  return _internal_has_custom_sub();
}
inline void TfLiteConverterCalculatorOptions::clear_custom_sub() {
  custom_sub_ = -1;
  _has_bits_[0] &= ~0x00000100u;
}
inline float TfLiteConverterCalculatorOptions::_internal_custom_sub() const {
  return custom_sub_;
}
inline float TfLiteConverterCalculatorOptions::custom_sub() const {
  // @@protoc_insertion_point(field_get:mediapipe.TfLiteConverterCalculatorOptions.custom_sub)
  return _internal_custom_sub();
}
inline void TfLiteConverterCalculatorOptions::_internal_set_custom_sub(float value) {
  _has_bits_[0] |= 0x00000100u;
  custom_sub_ = value;
}
inline void TfLiteConverterCalculatorOptions::set_custom_sub(float value) {
  _internal_set_custom_sub(value);
  // @@protoc_insertion_point(field_set:mediapipe.TfLiteConverterCalculatorOptions.custom_sub)
}

// optional bool flip_vertically = 2 [default = false];
inline bool TfLiteConverterCalculatorOptions::_internal_has_flip_vertically() const {
  bool value = (_has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool TfLiteConverterCalculatorOptions::has_flip_vertically() const {
  return _internal_has_flip_vertically();
}
inline void TfLiteConverterCalculatorOptions::clear_flip_vertically() {
  flip_vertically_ = false;
  _has_bits_[0] &= ~0x00000004u;
}
inline bool TfLiteConverterCalculatorOptions::_internal_flip_vertically() const {
  return flip_vertically_;
}
inline bool TfLiteConverterCalculatorOptions::flip_vertically() const {
  // @@protoc_insertion_point(field_get:mediapipe.TfLiteConverterCalculatorOptions.flip_vertically)
  return _internal_flip_vertically();
}
inline void TfLiteConverterCalculatorOptions::_internal_set_flip_vertically(bool value) {
  _has_bits_[0] |= 0x00000004u;
  flip_vertically_ = value;
}
inline void TfLiteConverterCalculatorOptions::set_flip_vertically(bool value) {
  _internal_set_flip_vertically(value);
  // @@protoc_insertion_point(field_set:mediapipe.TfLiteConverterCalculatorOptions.flip_vertically)
}

// optional int32 max_num_channels = 3 [default = 3];
inline bool TfLiteConverterCalculatorOptions::_internal_has_max_num_channels() const {
  bool value = (_has_bits_[0] & 0x00000040u) != 0;
  return value;
}
inline bool TfLiteConverterCalculatorOptions::has_max_num_channels() const {
  return _internal_has_max_num_channels();
}
inline void TfLiteConverterCalculatorOptions::clear_max_num_channels() {
  max_num_channels_ = 3;
  _has_bits_[0] &= ~0x00000040u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 TfLiteConverterCalculatorOptions::_internal_max_num_channels() const {
  return max_num_channels_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 TfLiteConverterCalculatorOptions::max_num_channels() const {
  // @@protoc_insertion_point(field_get:mediapipe.TfLiteConverterCalculatorOptions.max_num_channels)
  return _internal_max_num_channels();
}
inline void TfLiteConverterCalculatorOptions::_internal_set_max_num_channels(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000040u;
  max_num_channels_ = value;
}
inline void TfLiteConverterCalculatorOptions::set_max_num_channels(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_max_num_channels(value);
  // @@protoc_insertion_point(field_set:mediapipe.TfLiteConverterCalculatorOptions.max_num_channels)
}

// optional bool row_major_matrix = 4 [default = false];
inline bool TfLiteConverterCalculatorOptions::_internal_has_row_major_matrix() const {
  bool value = (_has_bits_[0] & 0x00000008u) != 0;
  return value;
}
inline bool TfLiteConverterCalculatorOptions::has_row_major_matrix() const {
  return _internal_has_row_major_matrix();
}
inline void TfLiteConverterCalculatorOptions::clear_row_major_matrix() {
  row_major_matrix_ = false;
  _has_bits_[0] &= ~0x00000008u;
}
inline bool TfLiteConverterCalculatorOptions::_internal_row_major_matrix() const {
  return row_major_matrix_;
}
inline bool TfLiteConverterCalculatorOptions::row_major_matrix() const {
  // @@protoc_insertion_point(field_get:mediapipe.TfLiteConverterCalculatorOptions.row_major_matrix)
  return _internal_row_major_matrix();
}
inline void TfLiteConverterCalculatorOptions::_internal_set_row_major_matrix(bool value) {
  _has_bits_[0] |= 0x00000008u;
  row_major_matrix_ = value;
}
inline void TfLiteConverterCalculatorOptions::set_row_major_matrix(bool value) {
  _internal_set_row_major_matrix(value);
  // @@protoc_insertion_point(field_set:mediapipe.TfLiteConverterCalculatorOptions.row_major_matrix)
}

// optional bool use_quantized_tensors = 5 [default = false];
inline bool TfLiteConverterCalculatorOptions::_internal_has_use_quantized_tensors() const {
  bool value = (_has_bits_[0] & 0x00000010u) != 0;
  return value;
}
inline bool TfLiteConverterCalculatorOptions::has_use_quantized_tensors() const {
  return _internal_has_use_quantized_tensors();
}
inline void TfLiteConverterCalculatorOptions::clear_use_quantized_tensors() {
  use_quantized_tensors_ = false;
  _has_bits_[0] &= ~0x00000010u;
}
inline bool TfLiteConverterCalculatorOptions::_internal_use_quantized_tensors() const {
  return use_quantized_tensors_;
}
inline bool TfLiteConverterCalculatorOptions::use_quantized_tensors() const {
  // @@protoc_insertion_point(field_get:mediapipe.TfLiteConverterCalculatorOptions.use_quantized_tensors)
  return _internal_use_quantized_tensors();
}
inline void TfLiteConverterCalculatorOptions::_internal_set_use_quantized_tensors(bool value) {
  _has_bits_[0] |= 0x00000010u;
  use_quantized_tensors_ = value;
}
inline void TfLiteConverterCalculatorOptions::set_use_quantized_tensors(bool value) {
  _internal_set_use_quantized_tensors(value);
  // @@protoc_insertion_point(field_set:mediapipe.TfLiteConverterCalculatorOptions.use_quantized_tensors)
}

// optional .mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange output_tensor_float_range = 9;
inline bool TfLiteConverterCalculatorOptions::_internal_has_output_tensor_float_range() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  PROTOBUF_ASSUME(!value || output_tensor_float_range_ != nullptr);
  return value;
}
inline bool TfLiteConverterCalculatorOptions::has_output_tensor_float_range() const {
  return _internal_has_output_tensor_float_range();
}
inline void TfLiteConverterCalculatorOptions::clear_output_tensor_float_range() {
  if (output_tensor_float_range_ != nullptr) output_tensor_float_range_->Clear();
  _has_bits_[0] &= ~0x00000001u;
}
inline const ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange& TfLiteConverterCalculatorOptions::_internal_output_tensor_float_range() const {
  const ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange* p = output_tensor_float_range_;
  return p != nullptr ? *p : reinterpret_cast<const ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange&>(
      ::mediapipe::_TfLiteConverterCalculatorOptions_TensorFloatRange_default_instance_);
}
inline const ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange& TfLiteConverterCalculatorOptions::output_tensor_float_range() const {
  // @@protoc_insertion_point(field_get:mediapipe.TfLiteConverterCalculatorOptions.output_tensor_float_range)
  return _internal_output_tensor_float_range();
}
inline void TfLiteConverterCalculatorOptions::unsafe_arena_set_allocated_output_tensor_float_range(
    ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange* output_tensor_float_range) {
  if (GetArena() == nullptr) {
    delete reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(output_tensor_float_range_);
  }
  output_tensor_float_range_ = output_tensor_float_range;
  if (output_tensor_float_range) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:mediapipe.TfLiteConverterCalculatorOptions.output_tensor_float_range)
}
inline ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange* TfLiteConverterCalculatorOptions::release_output_tensor_float_range() {
  _has_bits_[0] &= ~0x00000001u;
  ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange* temp = output_tensor_float_range_;
  output_tensor_float_range_ = nullptr;
  if (GetArena() != nullptr) {
    temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  }
  return temp;
}
inline ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange* TfLiteConverterCalculatorOptions::unsafe_arena_release_output_tensor_float_range() {
  // @@protoc_insertion_point(field_release:mediapipe.TfLiteConverterCalculatorOptions.output_tensor_float_range)
  _has_bits_[0] &= ~0x00000001u;
  ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange* temp = output_tensor_float_range_;
  output_tensor_float_range_ = nullptr;
  return temp;
}
inline ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange* TfLiteConverterCalculatorOptions::_internal_mutable_output_tensor_float_range() {
  _has_bits_[0] |= 0x00000001u;
  if (output_tensor_float_range_ == nullptr) {
    auto* p = CreateMaybeMessage<::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange>(GetArena());
    output_tensor_float_range_ = p;
  }
  return output_tensor_float_range_;
}
inline ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange* TfLiteConverterCalculatorOptions::mutable_output_tensor_float_range() {
  // @@protoc_insertion_point(field_mutable:mediapipe.TfLiteConverterCalculatorOptions.output_tensor_float_range)
  return _internal_mutable_output_tensor_float_range();
}
inline void TfLiteConverterCalculatorOptions::set_allocated_output_tensor_float_range(::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange* output_tensor_float_range) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArena();
  if (message_arena == nullptr) {
    delete output_tensor_float_range_;
  }
  if (output_tensor_float_range) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
      ::PROTOBUF_NAMESPACE_ID::Arena::GetArena(output_tensor_float_range);
    if (message_arena != submessage_arena) {
      output_tensor_float_range = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, output_tensor_float_range, submessage_arena);
    }
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  output_tensor_float_range_ = output_tensor_float_range;
  // @@protoc_insertion_point(field_set_allocated:mediapipe.TfLiteConverterCalculatorOptions.output_tensor_float_range)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <x/google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto
