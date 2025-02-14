// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/util/collection_has_min_size_calculator.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto;
namespace mediapipe {
class CollectionHasMinSizeCalculatorOptions;
struct CollectionHasMinSizeCalculatorOptionsDefaultTypeInternal;
extern CollectionHasMinSizeCalculatorOptionsDefaultTypeInternal _CollectionHasMinSizeCalculatorOptions_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::CollectionHasMinSizeCalculatorOptions* Arena::CreateMaybeMessage<::mediapipe::CollectionHasMinSizeCalculatorOptions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class CollectionHasMinSizeCalculatorOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.CollectionHasMinSizeCalculatorOptions) */ {
 public:
  inline CollectionHasMinSizeCalculatorOptions() : CollectionHasMinSizeCalculatorOptions(nullptr) {}
  ~CollectionHasMinSizeCalculatorOptions() override;
  explicit constexpr CollectionHasMinSizeCalculatorOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  CollectionHasMinSizeCalculatorOptions(const CollectionHasMinSizeCalculatorOptions& from);
  CollectionHasMinSizeCalculatorOptions(CollectionHasMinSizeCalculatorOptions&& from) noexcept
    : CollectionHasMinSizeCalculatorOptions() {
    *this = ::std::move(from);
  }

  inline CollectionHasMinSizeCalculatorOptions& operator=(const CollectionHasMinSizeCalculatorOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline CollectionHasMinSizeCalculatorOptions& operator=(CollectionHasMinSizeCalculatorOptions&& from) noexcept {
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
  static const CollectionHasMinSizeCalculatorOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const CollectionHasMinSizeCalculatorOptions* internal_default_instance() {
    return reinterpret_cast<const CollectionHasMinSizeCalculatorOptions*>(
               &_CollectionHasMinSizeCalculatorOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(CollectionHasMinSizeCalculatorOptions& a, CollectionHasMinSizeCalculatorOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(CollectionHasMinSizeCalculatorOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(CollectionHasMinSizeCalculatorOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline CollectionHasMinSizeCalculatorOptions* New() const final {
    return CreateMaybeMessage<CollectionHasMinSizeCalculatorOptions>(nullptr);
  }

  CollectionHasMinSizeCalculatorOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<CollectionHasMinSizeCalculatorOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const CollectionHasMinSizeCalculatorOptions& from);
  void MergeFrom(const CollectionHasMinSizeCalculatorOptions& from);
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
  void InternalSwap(CollectionHasMinSizeCalculatorOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.CollectionHasMinSizeCalculatorOptions";
  }
  protected:
  explicit CollectionHasMinSizeCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kMinSizeFieldNumber = 1,
  };
  // optional int32 min_size = 1 [default = 0];
  bool has_min_size() const;
  private:
  bool _internal_has_min_size() const;
  public:
  void clear_min_size();
  ::PROTOBUF_NAMESPACE_ID::int32 min_size() const;
  void set_min_size(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_min_size() const;
  void _internal_set_min_size(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  static const int kExtFieldNumber = 259397840;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::CollectionHasMinSizeCalculatorOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.CollectionHasMinSizeCalculatorOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::int32 min_size_;
  friend struct ::TableStruct_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// CollectionHasMinSizeCalculatorOptions

// optional int32 min_size = 1 [default = 0];
inline bool CollectionHasMinSizeCalculatorOptions::_internal_has_min_size() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool CollectionHasMinSizeCalculatorOptions::has_min_size() const {
  return _internal_has_min_size();
}
inline void CollectionHasMinSizeCalculatorOptions::clear_min_size() {
  min_size_ = 0;
  _has_bits_[0] &= ~0x00000001u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 CollectionHasMinSizeCalculatorOptions::_internal_min_size() const {
  return min_size_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 CollectionHasMinSizeCalculatorOptions::min_size() const {
  // @@protoc_insertion_point(field_get:mediapipe.CollectionHasMinSizeCalculatorOptions.min_size)
  return _internal_min_size();
}
inline void CollectionHasMinSizeCalculatorOptions::_internal_set_min_size(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000001u;
  min_size_ = value;
}
inline void CollectionHasMinSizeCalculatorOptions::set_min_size(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_min_size(value);
  // @@protoc_insertion_point(field_set:mediapipe.CollectionHasMinSizeCalculatorOptions.min_size)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2futil_2fcollection_5fhas_5fmin_5fsize_5fcalculator_2eproto
