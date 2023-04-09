// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/core/split_vector_calculator.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto;
namespace mediapipe {
class Range;
struct RangeDefaultTypeInternal;
extern RangeDefaultTypeInternal _Range_default_instance_;
class SplitVectorCalculatorOptions;
struct SplitVectorCalculatorOptionsDefaultTypeInternal;
extern SplitVectorCalculatorOptionsDefaultTypeInternal _SplitVectorCalculatorOptions_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::Range* Arena::CreateMaybeMessage<::mediapipe::Range>(Arena*);
template<> ::mediapipe::SplitVectorCalculatorOptions* Arena::CreateMaybeMessage<::mediapipe::SplitVectorCalculatorOptions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class Range PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.Range) */ {
 public:
  inline Range() : Range(nullptr) {}
  ~Range() override;
  explicit constexpr Range(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  Range(const Range& from);
  Range(Range&& from) noexcept
    : Range() {
    *this = ::std::move(from);
  }

  inline Range& operator=(const Range& from) {
    CopyFrom(from);
    return *this;
  }
  inline Range& operator=(Range&& from) noexcept {
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
  static const Range& default_instance() {
    return *internal_default_instance();
  }
  static inline const Range* internal_default_instance() {
    return reinterpret_cast<const Range*>(
               &_Range_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(Range& a, Range& b) {
    a.Swap(&b);
  }
  inline void Swap(Range* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(Range* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline Range* New() const final {
    return CreateMaybeMessage<Range>(nullptr);
  }

  Range* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<Range>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const Range& from);
  void MergeFrom(const Range& from);
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
  void InternalSwap(Range* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.Range";
  }
  protected:
  explicit Range(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kBeginFieldNumber = 1,
    kEndFieldNumber = 2,
  };
  // optional int32 begin = 1;
  bool has_begin() const;
  private:
  bool _internal_has_begin() const;
  public:
  void clear_begin();
  ::PROTOBUF_NAMESPACE_ID::int32 begin() const;
  void set_begin(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_begin() const;
  void _internal_set_begin(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional int32 end = 2;
  bool has_end() const;
  private:
  bool _internal_has_end() const;
  public:
  void clear_end();
  ::PROTOBUF_NAMESPACE_ID::int32 end() const;
  void set_end(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_end() const;
  void _internal_set_end(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // @@protoc_insertion_point(class_scope:mediapipe.Range)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::int32 begin_;
  ::PROTOBUF_NAMESPACE_ID::int32 end_;
  friend struct ::TableStruct_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto;
};
// -------------------------------------------------------------------

class SplitVectorCalculatorOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.SplitVectorCalculatorOptions) */ {
 public:
  inline SplitVectorCalculatorOptions() : SplitVectorCalculatorOptions(nullptr) {}
  ~SplitVectorCalculatorOptions() override;
  explicit constexpr SplitVectorCalculatorOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  SplitVectorCalculatorOptions(const SplitVectorCalculatorOptions& from);
  SplitVectorCalculatorOptions(SplitVectorCalculatorOptions&& from) noexcept
    : SplitVectorCalculatorOptions() {
    *this = ::std::move(from);
  }

  inline SplitVectorCalculatorOptions& operator=(const SplitVectorCalculatorOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline SplitVectorCalculatorOptions& operator=(SplitVectorCalculatorOptions&& from) noexcept {
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
  static const SplitVectorCalculatorOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const SplitVectorCalculatorOptions* internal_default_instance() {
    return reinterpret_cast<const SplitVectorCalculatorOptions*>(
               &_SplitVectorCalculatorOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(SplitVectorCalculatorOptions& a, SplitVectorCalculatorOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(SplitVectorCalculatorOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(SplitVectorCalculatorOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline SplitVectorCalculatorOptions* New() const final {
    return CreateMaybeMessage<SplitVectorCalculatorOptions>(nullptr);
  }

  SplitVectorCalculatorOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<SplitVectorCalculatorOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const SplitVectorCalculatorOptions& from);
  void MergeFrom(const SplitVectorCalculatorOptions& from);
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
  void InternalSwap(SplitVectorCalculatorOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.SplitVectorCalculatorOptions";
  }
  protected:
  explicit SplitVectorCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kRangesFieldNumber = 1,
    kElementOnlyFieldNumber = 2,
    kCombineOutputsFieldNumber = 3,
  };
  // repeated .mediapipe.Range ranges = 1;
  int ranges_size() const;
  private:
  int _internal_ranges_size() const;
  public:
  void clear_ranges();
  ::mediapipe::Range* mutable_ranges(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::Range >*
      mutable_ranges();
  private:
  const ::mediapipe::Range& _internal_ranges(int index) const;
  ::mediapipe::Range* _internal_add_ranges();
  public:
  const ::mediapipe::Range& ranges(int index) const;
  ::mediapipe::Range* add_ranges();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::Range >&
      ranges() const;

  // optional bool element_only = 2 [default = false];
  bool has_element_only() const;
  private:
  bool _internal_has_element_only() const;
  public:
  void clear_element_only();
  bool element_only() const;
  void set_element_only(bool value);
  private:
  bool _internal_element_only() const;
  void _internal_set_element_only(bool value);
  public:

  // optional bool combine_outputs = 3 [default = false];
  bool has_combine_outputs() const;
  private:
  bool _internal_has_combine_outputs() const;
  public:
  void clear_combine_outputs();
  bool combine_outputs() const;
  void set_combine_outputs(bool value);
  private:
  bool _internal_combine_outputs() const;
  void _internal_set_combine_outputs(bool value);
  public:

  static const int kExtFieldNumber = 259438222;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::SplitVectorCalculatorOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.SplitVectorCalculatorOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::Range > ranges_;
  bool element_only_;
  bool combine_outputs_;
  friend struct ::TableStruct_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// Range

// optional int32 begin = 1;
inline bool Range::_internal_has_begin() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool Range::has_begin() const {
  return _internal_has_begin();
}
inline void Range::clear_begin() {
  begin_ = 0;
  _has_bits_[0] &= ~0x00000001u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Range::_internal_begin() const {
  return begin_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Range::begin() const {
  // @@protoc_insertion_point(field_get:mediapipe.Range.begin)
  return _internal_begin();
}
inline void Range::_internal_set_begin(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000001u;
  begin_ = value;
}
inline void Range::set_begin(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_begin(value);
  // @@protoc_insertion_point(field_set:mediapipe.Range.begin)
}

// optional int32 end = 2;
inline bool Range::_internal_has_end() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool Range::has_end() const {
  return _internal_has_end();
}
inline void Range::clear_end() {
  end_ = 0;
  _has_bits_[0] &= ~0x00000002u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Range::_internal_end() const {
  return end_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Range::end() const {
  // @@protoc_insertion_point(field_get:mediapipe.Range.end)
  return _internal_end();
}
inline void Range::_internal_set_end(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000002u;
  end_ = value;
}
inline void Range::set_end(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_end(value);
  // @@protoc_insertion_point(field_set:mediapipe.Range.end)
}

// -------------------------------------------------------------------

// SplitVectorCalculatorOptions

// repeated .mediapipe.Range ranges = 1;
inline int SplitVectorCalculatorOptions::_internal_ranges_size() const {
  return ranges_.size();
}
inline int SplitVectorCalculatorOptions::ranges_size() const {
  return _internal_ranges_size();
}
inline void SplitVectorCalculatorOptions::clear_ranges() {
  ranges_.Clear();
}
inline ::mediapipe::Range* SplitVectorCalculatorOptions::mutable_ranges(int index) {
  // @@protoc_insertion_point(field_mutable:mediapipe.SplitVectorCalculatorOptions.ranges)
  return ranges_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::Range >*
SplitVectorCalculatorOptions::mutable_ranges() {
  // @@protoc_insertion_point(field_mutable_list:mediapipe.SplitVectorCalculatorOptions.ranges)
  return &ranges_;
}
inline const ::mediapipe::Range& SplitVectorCalculatorOptions::_internal_ranges(int index) const {
  return ranges_.Get(index);
}
inline const ::mediapipe::Range& SplitVectorCalculatorOptions::ranges(int index) const {
  // @@protoc_insertion_point(field_get:mediapipe.SplitVectorCalculatorOptions.ranges)
  return _internal_ranges(index);
}
inline ::mediapipe::Range* SplitVectorCalculatorOptions::_internal_add_ranges() {
  return ranges_.Add();
}
inline ::mediapipe::Range* SplitVectorCalculatorOptions::add_ranges() {
  // @@protoc_insertion_point(field_add:mediapipe.SplitVectorCalculatorOptions.ranges)
  return _internal_add_ranges();
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::Range >&
SplitVectorCalculatorOptions::ranges() const {
  // @@protoc_insertion_point(field_list:mediapipe.SplitVectorCalculatorOptions.ranges)
  return ranges_;
}

// optional bool element_only = 2 [default = false];
inline bool SplitVectorCalculatorOptions::_internal_has_element_only() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool SplitVectorCalculatorOptions::has_element_only() const {
  return _internal_has_element_only();
}
inline void SplitVectorCalculatorOptions::clear_element_only() {
  element_only_ = false;
  _has_bits_[0] &= ~0x00000001u;
}
inline bool SplitVectorCalculatorOptions::_internal_element_only() const {
  return element_only_;
}
inline bool SplitVectorCalculatorOptions::element_only() const {
  // @@protoc_insertion_point(field_get:mediapipe.SplitVectorCalculatorOptions.element_only)
  return _internal_element_only();
}
inline void SplitVectorCalculatorOptions::_internal_set_element_only(bool value) {
  _has_bits_[0] |= 0x00000001u;
  element_only_ = value;
}
inline void SplitVectorCalculatorOptions::set_element_only(bool value) {
  _internal_set_element_only(value);
  // @@protoc_insertion_point(field_set:mediapipe.SplitVectorCalculatorOptions.element_only)
}

// optional bool combine_outputs = 3 [default = false];
inline bool SplitVectorCalculatorOptions::_internal_has_combine_outputs() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool SplitVectorCalculatorOptions::has_combine_outputs() const {
  return _internal_has_combine_outputs();
}
inline void SplitVectorCalculatorOptions::clear_combine_outputs() {
  combine_outputs_ = false;
  _has_bits_[0] &= ~0x00000002u;
}
inline bool SplitVectorCalculatorOptions::_internal_combine_outputs() const {
  return combine_outputs_;
}
inline bool SplitVectorCalculatorOptions::combine_outputs() const {
  // @@protoc_insertion_point(field_get:mediapipe.SplitVectorCalculatorOptions.combine_outputs)
  return _internal_combine_outputs();
}
inline void SplitVectorCalculatorOptions::_internal_set_combine_outputs(bool value) {
  _has_bits_[0] |= 0x00000002u;
  combine_outputs_ = value;
}
inline void SplitVectorCalculatorOptions::set_combine_outputs(bool value) {
  _internal_set_combine_outputs(value);
  // @@protoc_insertion_point(field_set:mediapipe.SplitVectorCalculatorOptions.combine_outputs)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <x/google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto
