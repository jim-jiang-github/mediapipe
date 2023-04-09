// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/video/box_detector_calculator.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fvideo_2fbox_5fdetector_5fcalculator_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fvideo_2fbox_5fdetector_5fcalculator_2eproto

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
#include "mediapipe/util/tracking/box_detector.pb.h"
// @@protoc_insertion_point(includes)
#include <x/google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fcalculators_2fvideo_2fbox_5fdetector_5fcalculator_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fcalculators_2fvideo_2fbox_5fdetector_5fcalculator_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2fvideo_2fbox_5fdetector_5fcalculator_2eproto;
namespace mediapipe {
class BoxDetectorCalculatorOptions;
struct BoxDetectorCalculatorOptionsDefaultTypeInternal;
extern BoxDetectorCalculatorOptionsDefaultTypeInternal _BoxDetectorCalculatorOptions_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::BoxDetectorCalculatorOptions* Arena::CreateMaybeMessage<::mediapipe::BoxDetectorCalculatorOptions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class BoxDetectorCalculatorOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.BoxDetectorCalculatorOptions) */ {
 public:
  inline BoxDetectorCalculatorOptions() : BoxDetectorCalculatorOptions(nullptr) {}
  ~BoxDetectorCalculatorOptions() override;
  explicit constexpr BoxDetectorCalculatorOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  BoxDetectorCalculatorOptions(const BoxDetectorCalculatorOptions& from);
  BoxDetectorCalculatorOptions(BoxDetectorCalculatorOptions&& from) noexcept
    : BoxDetectorCalculatorOptions() {
    *this = ::std::move(from);
  }

  inline BoxDetectorCalculatorOptions& operator=(const BoxDetectorCalculatorOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline BoxDetectorCalculatorOptions& operator=(BoxDetectorCalculatorOptions&& from) noexcept {
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
  static const BoxDetectorCalculatorOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const BoxDetectorCalculatorOptions* internal_default_instance() {
    return reinterpret_cast<const BoxDetectorCalculatorOptions*>(
               &_BoxDetectorCalculatorOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(BoxDetectorCalculatorOptions& a, BoxDetectorCalculatorOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(BoxDetectorCalculatorOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(BoxDetectorCalculatorOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline BoxDetectorCalculatorOptions* New() const final {
    return CreateMaybeMessage<BoxDetectorCalculatorOptions>(nullptr);
  }

  BoxDetectorCalculatorOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<BoxDetectorCalculatorOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const BoxDetectorCalculatorOptions& from);
  void MergeFrom(const BoxDetectorCalculatorOptions& from);
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
  void InternalSwap(BoxDetectorCalculatorOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.BoxDetectorCalculatorOptions";
  }
  protected:
  explicit BoxDetectorCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kIndexProtoFilenameFieldNumber = 2,
    kDetectorOptionsFieldNumber = 1,
  };
  // repeated string index_proto_filename = 2;
  int index_proto_filename_size() const;
  private:
  int _internal_index_proto_filename_size() const;
  public:
  void clear_index_proto_filename();
  const std::string& index_proto_filename(int index) const;
  std::string* mutable_index_proto_filename(int index);
  void set_index_proto_filename(int index, const std::string& value);
  void set_index_proto_filename(int index, std::string&& value);
  void set_index_proto_filename(int index, const char* value);
  void set_index_proto_filename(int index, const char* value, size_t size);
  std::string* add_index_proto_filename();
  void add_index_proto_filename(const std::string& value);
  void add_index_proto_filename(std::string&& value);
  void add_index_proto_filename(const char* value);
  void add_index_proto_filename(const char* value, size_t size);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>& index_proto_filename() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>* mutable_index_proto_filename();
  private:
  const std::string& _internal_index_proto_filename(int index) const;
  std::string* _internal_add_index_proto_filename();
  public:

  // optional .mediapipe.BoxDetectorOptions detector_options = 1;
  bool has_detector_options() const;
  private:
  bool _internal_has_detector_options() const;
  public:
  void clear_detector_options();
  const ::mediapipe::BoxDetectorOptions& detector_options() const;
  PROTOBUF_FUTURE_MUST_USE_RESULT ::mediapipe::BoxDetectorOptions* release_detector_options();
  ::mediapipe::BoxDetectorOptions* mutable_detector_options();
  void set_allocated_detector_options(::mediapipe::BoxDetectorOptions* detector_options);
  private:
  const ::mediapipe::BoxDetectorOptions& _internal_detector_options() const;
  ::mediapipe::BoxDetectorOptions* _internal_mutable_detector_options();
  public:
  void unsafe_arena_set_allocated_detector_options(
      ::mediapipe::BoxDetectorOptions* detector_options);
  ::mediapipe::BoxDetectorOptions* unsafe_arena_release_detector_options();

  static const int kExtFieldNumber = 289746530;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::BoxDetectorCalculatorOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.BoxDetectorCalculatorOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string> index_proto_filename_;
  ::mediapipe::BoxDetectorOptions* detector_options_;
  friend struct ::TableStruct_mediapipe_2fcalculators_2fvideo_2fbox_5fdetector_5fcalculator_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// BoxDetectorCalculatorOptions

// optional .mediapipe.BoxDetectorOptions detector_options = 1;
inline bool BoxDetectorCalculatorOptions::_internal_has_detector_options() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  PROTOBUF_ASSUME(!value || detector_options_ != nullptr);
  return value;
}
inline bool BoxDetectorCalculatorOptions::has_detector_options() const {
  return _internal_has_detector_options();
}
inline const ::mediapipe::BoxDetectorOptions& BoxDetectorCalculatorOptions::_internal_detector_options() const {
  const ::mediapipe::BoxDetectorOptions* p = detector_options_;
  return p != nullptr ? *p : reinterpret_cast<const ::mediapipe::BoxDetectorOptions&>(
      ::mediapipe::_BoxDetectorOptions_default_instance_);
}
inline const ::mediapipe::BoxDetectorOptions& BoxDetectorCalculatorOptions::detector_options() const {
  // @@protoc_insertion_point(field_get:mediapipe.BoxDetectorCalculatorOptions.detector_options)
  return _internal_detector_options();
}
inline void BoxDetectorCalculatorOptions::unsafe_arena_set_allocated_detector_options(
    ::mediapipe::BoxDetectorOptions* detector_options) {
  if (GetArena() == nullptr) {
    delete reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(detector_options_);
  }
  detector_options_ = detector_options;
  if (detector_options) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:mediapipe.BoxDetectorCalculatorOptions.detector_options)
}
inline ::mediapipe::BoxDetectorOptions* BoxDetectorCalculatorOptions::release_detector_options() {
  _has_bits_[0] &= ~0x00000001u;
  ::mediapipe::BoxDetectorOptions* temp = detector_options_;
  detector_options_ = nullptr;
  if (GetArena() != nullptr) {
    temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  }
  return temp;
}
inline ::mediapipe::BoxDetectorOptions* BoxDetectorCalculatorOptions::unsafe_arena_release_detector_options() {
  // @@protoc_insertion_point(field_release:mediapipe.BoxDetectorCalculatorOptions.detector_options)
  _has_bits_[0] &= ~0x00000001u;
  ::mediapipe::BoxDetectorOptions* temp = detector_options_;
  detector_options_ = nullptr;
  return temp;
}
inline ::mediapipe::BoxDetectorOptions* BoxDetectorCalculatorOptions::_internal_mutable_detector_options() {
  _has_bits_[0] |= 0x00000001u;
  if (detector_options_ == nullptr) {
    auto* p = CreateMaybeMessage<::mediapipe::BoxDetectorOptions>(GetArena());
    detector_options_ = p;
  }
  return detector_options_;
}
inline ::mediapipe::BoxDetectorOptions* BoxDetectorCalculatorOptions::mutable_detector_options() {
  // @@protoc_insertion_point(field_mutable:mediapipe.BoxDetectorCalculatorOptions.detector_options)
  return _internal_mutable_detector_options();
}
inline void BoxDetectorCalculatorOptions::set_allocated_detector_options(::mediapipe::BoxDetectorOptions* detector_options) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArena();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(detector_options_);
  }
  if (detector_options) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
      reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(detector_options)->GetArena();
    if (message_arena != submessage_arena) {
      detector_options = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, detector_options, submessage_arena);
    }
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  detector_options_ = detector_options;
  // @@protoc_insertion_point(field_set_allocated:mediapipe.BoxDetectorCalculatorOptions.detector_options)
}

// repeated string index_proto_filename = 2;
inline int BoxDetectorCalculatorOptions::_internal_index_proto_filename_size() const {
  return index_proto_filename_.size();
}
inline int BoxDetectorCalculatorOptions::index_proto_filename_size() const {
  return _internal_index_proto_filename_size();
}
inline void BoxDetectorCalculatorOptions::clear_index_proto_filename() {
  index_proto_filename_.Clear();
}
inline std::string* BoxDetectorCalculatorOptions::add_index_proto_filename() {
  // @@protoc_insertion_point(field_add_mutable:mediapipe.BoxDetectorCalculatorOptions.index_proto_filename)
  return _internal_add_index_proto_filename();
}
inline const std::string& BoxDetectorCalculatorOptions::_internal_index_proto_filename(int index) const {
  return index_proto_filename_.Get(index);
}
inline const std::string& BoxDetectorCalculatorOptions::index_proto_filename(int index) const {
  // @@protoc_insertion_point(field_get:mediapipe.BoxDetectorCalculatorOptions.index_proto_filename)
  return _internal_index_proto_filename(index);
}
inline std::string* BoxDetectorCalculatorOptions::mutable_index_proto_filename(int index) {
  // @@protoc_insertion_point(field_mutable:mediapipe.BoxDetectorCalculatorOptions.index_proto_filename)
  return index_proto_filename_.Mutable(index);
}
inline void BoxDetectorCalculatorOptions::set_index_proto_filename(int index, const std::string& value) {
  // @@protoc_insertion_point(field_set:mediapipe.BoxDetectorCalculatorOptions.index_proto_filename)
  index_proto_filename_.Mutable(index)->assign(value);
}
inline void BoxDetectorCalculatorOptions::set_index_proto_filename(int index, std::string&& value) {
  // @@protoc_insertion_point(field_set:mediapipe.BoxDetectorCalculatorOptions.index_proto_filename)
  index_proto_filename_.Mutable(index)->assign(std::move(value));
}
inline void BoxDetectorCalculatorOptions::set_index_proto_filename(int index, const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  index_proto_filename_.Mutable(index)->assign(value);
  // @@protoc_insertion_point(field_set_char:mediapipe.BoxDetectorCalculatorOptions.index_proto_filename)
}
inline void BoxDetectorCalculatorOptions::set_index_proto_filename(int index, const char* value, size_t size) {
  index_proto_filename_.Mutable(index)->assign(
    reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:mediapipe.BoxDetectorCalculatorOptions.index_proto_filename)
}
inline std::string* BoxDetectorCalculatorOptions::_internal_add_index_proto_filename() {
  return index_proto_filename_.Add();
}
inline void BoxDetectorCalculatorOptions::add_index_proto_filename(const std::string& value) {
  index_proto_filename_.Add()->assign(value);
  // @@protoc_insertion_point(field_add:mediapipe.BoxDetectorCalculatorOptions.index_proto_filename)
}
inline void BoxDetectorCalculatorOptions::add_index_proto_filename(std::string&& value) {
  index_proto_filename_.Add(std::move(value));
  // @@protoc_insertion_point(field_add:mediapipe.BoxDetectorCalculatorOptions.index_proto_filename)
}
inline void BoxDetectorCalculatorOptions::add_index_proto_filename(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  index_proto_filename_.Add()->assign(value);
  // @@protoc_insertion_point(field_add_char:mediapipe.BoxDetectorCalculatorOptions.index_proto_filename)
}
inline void BoxDetectorCalculatorOptions::add_index_proto_filename(const char* value, size_t size) {
  index_proto_filename_.Add()->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_add_pointer:mediapipe.BoxDetectorCalculatorOptions.index_proto_filename)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>&
BoxDetectorCalculatorOptions::index_proto_filename() const {
  // @@protoc_insertion_point(field_list:mediapipe.BoxDetectorCalculatorOptions.index_proto_filename)
  return index_proto_filename_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>*
BoxDetectorCalculatorOptions::mutable_index_proto_filename() {
  // @@protoc_insertion_point(field_mutable_list:mediapipe.BoxDetectorCalculatorOptions.index_proto_filename)
  return &index_proto_filename_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <x/google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fvideo_2fbox_5fdetector_5fcalculator_2eproto
