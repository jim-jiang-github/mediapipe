// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/util/detection_label_id_to_text_calculator.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto

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
#include <x/google/protobuf/map.h>  // IWYU pragma: export
#include <x/google/protobuf/map_entry.h>
#include <x/google/protobuf/map_field_inl.h>
#include <x/google/protobuf/unknown_field_set.h>
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/util/label_map.pb.h"
// @@protoc_insertion_point(includes)
#include <x/google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto;
namespace mediapipe {
class DetectionLabelIdToTextCalculatorOptions;
struct DetectionLabelIdToTextCalculatorOptionsDefaultTypeInternal;
extern DetectionLabelIdToTextCalculatorOptionsDefaultTypeInternal _DetectionLabelIdToTextCalculatorOptions_default_instance_;
class DetectionLabelIdToTextCalculatorOptions_LabelItemsEntry_DoNotUse;
struct DetectionLabelIdToTextCalculatorOptions_LabelItemsEntry_DoNotUseDefaultTypeInternal;
extern DetectionLabelIdToTextCalculatorOptions_LabelItemsEntry_DoNotUseDefaultTypeInternal _DetectionLabelIdToTextCalculatorOptions_LabelItemsEntry_DoNotUse_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::DetectionLabelIdToTextCalculatorOptions* Arena::CreateMaybeMessage<::mediapipe::DetectionLabelIdToTextCalculatorOptions>(Arena*);
template<> ::mediapipe::DetectionLabelIdToTextCalculatorOptions_LabelItemsEntry_DoNotUse* Arena::CreateMaybeMessage<::mediapipe::DetectionLabelIdToTextCalculatorOptions_LabelItemsEntry_DoNotUse>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class DetectionLabelIdToTextCalculatorOptions_LabelItemsEntry_DoNotUse : public ::PROTOBUF_NAMESPACE_ID::internal::MapEntry<DetectionLabelIdToTextCalculatorOptions_LabelItemsEntry_DoNotUse, 
    ::PROTOBUF_NAMESPACE_ID::int64, ::mediapipe::LabelMapItem,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT64,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_MESSAGE> {
public:
  typedef ::PROTOBUF_NAMESPACE_ID::internal::MapEntry<DetectionLabelIdToTextCalculatorOptions_LabelItemsEntry_DoNotUse, 
    ::PROTOBUF_NAMESPACE_ID::int64, ::mediapipe::LabelMapItem,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT64,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_MESSAGE> SuperType;
  DetectionLabelIdToTextCalculatorOptions_LabelItemsEntry_DoNotUse();
  explicit constexpr DetectionLabelIdToTextCalculatorOptions_LabelItemsEntry_DoNotUse(
      ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);
  explicit DetectionLabelIdToTextCalculatorOptions_LabelItemsEntry_DoNotUse(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  void MergeFrom(const DetectionLabelIdToTextCalculatorOptions_LabelItemsEntry_DoNotUse& other);
  static const DetectionLabelIdToTextCalculatorOptions_LabelItemsEntry_DoNotUse* internal_default_instance() { return reinterpret_cast<const DetectionLabelIdToTextCalculatorOptions_LabelItemsEntry_DoNotUse*>(&_DetectionLabelIdToTextCalculatorOptions_LabelItemsEntry_DoNotUse_default_instance_); }
  static bool ValidateKey(void*) { return true; }
  static bool ValidateValue(void*) { return true; }
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& other) final;
  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
};

// -------------------------------------------------------------------

class DetectionLabelIdToTextCalculatorOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.DetectionLabelIdToTextCalculatorOptions) */ {
 public:
  inline DetectionLabelIdToTextCalculatorOptions() : DetectionLabelIdToTextCalculatorOptions(nullptr) {}
  ~DetectionLabelIdToTextCalculatorOptions() override;
  explicit constexpr DetectionLabelIdToTextCalculatorOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  DetectionLabelIdToTextCalculatorOptions(const DetectionLabelIdToTextCalculatorOptions& from);
  DetectionLabelIdToTextCalculatorOptions(DetectionLabelIdToTextCalculatorOptions&& from) noexcept
    : DetectionLabelIdToTextCalculatorOptions() {
    *this = ::std::move(from);
  }

  inline DetectionLabelIdToTextCalculatorOptions& operator=(const DetectionLabelIdToTextCalculatorOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline DetectionLabelIdToTextCalculatorOptions& operator=(DetectionLabelIdToTextCalculatorOptions&& from) noexcept {
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
  static const DetectionLabelIdToTextCalculatorOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const DetectionLabelIdToTextCalculatorOptions* internal_default_instance() {
    return reinterpret_cast<const DetectionLabelIdToTextCalculatorOptions*>(
               &_DetectionLabelIdToTextCalculatorOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(DetectionLabelIdToTextCalculatorOptions& a, DetectionLabelIdToTextCalculatorOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(DetectionLabelIdToTextCalculatorOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(DetectionLabelIdToTextCalculatorOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline DetectionLabelIdToTextCalculatorOptions* New() const final {
    return CreateMaybeMessage<DetectionLabelIdToTextCalculatorOptions>(nullptr);
  }

  DetectionLabelIdToTextCalculatorOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<DetectionLabelIdToTextCalculatorOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const DetectionLabelIdToTextCalculatorOptions& from);
  void MergeFrom(const DetectionLabelIdToTextCalculatorOptions& from);
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
  void InternalSwap(DetectionLabelIdToTextCalculatorOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.DetectionLabelIdToTextCalculatorOptions";
  }
  protected:
  explicit DetectionLabelIdToTextCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------


  // accessors -------------------------------------------------------

  enum : int {
    kLabelFieldNumber = 2,
    kLabelItemsFieldNumber = 4,
    kLabelMapPathFieldNumber = 1,
    kKeepLabelIdFieldNumber = 3,
  };
  // repeated string label = 2;
  int label_size() const;
  private:
  int _internal_label_size() const;
  public:
  void clear_label();
  const std::string& label(int index) const;
  std::string* mutable_label(int index);
  void set_label(int index, const std::string& value);
  void set_label(int index, std::string&& value);
  void set_label(int index, const char* value);
  void set_label(int index, const char* value, size_t size);
  std::string* add_label();
  void add_label(const std::string& value);
  void add_label(std::string&& value);
  void add_label(const char* value);
  void add_label(const char* value, size_t size);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>& label() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>* mutable_label();
  private:
  const std::string& _internal_label(int index) const;
  std::string* _internal_add_label();
  public:

  // map<int64, .mediapipe.LabelMapItem> label_items = 4;
  int label_items_size() const;
  private:
  int _internal_label_items_size() const;
  public:
  void clear_label_items();
  private:
  const ::PROTOBUF_NAMESPACE_ID::Map< ::PROTOBUF_NAMESPACE_ID::int64, ::mediapipe::LabelMapItem >&
      _internal_label_items() const;
  ::PROTOBUF_NAMESPACE_ID::Map< ::PROTOBUF_NAMESPACE_ID::int64, ::mediapipe::LabelMapItem >*
      _internal_mutable_label_items();
  public:
  const ::PROTOBUF_NAMESPACE_ID::Map< ::PROTOBUF_NAMESPACE_ID::int64, ::mediapipe::LabelMapItem >&
      label_items() const;
  ::PROTOBUF_NAMESPACE_ID::Map< ::PROTOBUF_NAMESPACE_ID::int64, ::mediapipe::LabelMapItem >*
      mutable_label_items();

  // optional string label_map_path = 1;
  bool has_label_map_path() const;
  private:
  bool _internal_has_label_map_path() const;
  public:
  void clear_label_map_path();
  const std::string& label_map_path() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_label_map_path(ArgT0&& arg0, ArgT... args);
  std::string* mutable_label_map_path();
  std::string* release_label_map_path();
  void set_allocated_label_map_path(std::string* label_map_path);
  private:
  const std::string& _internal_label_map_path() const;
  void _internal_set_label_map_path(const std::string& value);
  std::string* _internal_mutable_label_map_path();
  public:

  // optional bool keep_label_id = 3;
  bool has_keep_label_id() const;
  private:
  bool _internal_has_keep_label_id() const;
  public:
  void clear_keep_label_id();
  bool keep_label_id() const;
  void set_keep_label_id(bool value);
  private:
  bool _internal_keep_label_id() const;
  void _internal_set_keep_label_id(bool value);
  public:

  static const int kExtFieldNumber = 251889072;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::DetectionLabelIdToTextCalculatorOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.DetectionLabelIdToTextCalculatorOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string> label_;
  ::PROTOBUF_NAMESPACE_ID::internal::MapField<
      DetectionLabelIdToTextCalculatorOptions_LabelItemsEntry_DoNotUse,
      ::PROTOBUF_NAMESPACE_ID::int64, ::mediapipe::LabelMapItem,
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT64,
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_MESSAGE> label_items_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr label_map_path_;
  bool keep_label_id_;
  friend struct ::TableStruct_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// -------------------------------------------------------------------

// DetectionLabelIdToTextCalculatorOptions

// optional string label_map_path = 1;
inline bool DetectionLabelIdToTextCalculatorOptions::_internal_has_label_map_path() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool DetectionLabelIdToTextCalculatorOptions::has_label_map_path() const {
  return _internal_has_label_map_path();
}
inline void DetectionLabelIdToTextCalculatorOptions::clear_label_map_path() {
  label_map_path_.ClearToEmpty();
  _has_bits_[0] &= ~0x00000001u;
}
inline const std::string& DetectionLabelIdToTextCalculatorOptions::label_map_path() const {
  // @@protoc_insertion_point(field_get:mediapipe.DetectionLabelIdToTextCalculatorOptions.label_map_path)
  return _internal_label_map_path();
}
template <typename ArgT0, typename... ArgT>
PROTOBUF_ALWAYS_INLINE
inline void DetectionLabelIdToTextCalculatorOptions::set_label_map_path(ArgT0&& arg0, ArgT... args) {
 _has_bits_[0] |= 0x00000001u;
 label_map_path_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArena());
  // @@protoc_insertion_point(field_set:mediapipe.DetectionLabelIdToTextCalculatorOptions.label_map_path)
}
inline std::string* DetectionLabelIdToTextCalculatorOptions::mutable_label_map_path() {
  // @@protoc_insertion_point(field_mutable:mediapipe.DetectionLabelIdToTextCalculatorOptions.label_map_path)
  return _internal_mutable_label_map_path();
}
inline const std::string& DetectionLabelIdToTextCalculatorOptions::_internal_label_map_path() const {
  return label_map_path_.Get();
}
inline void DetectionLabelIdToTextCalculatorOptions::_internal_set_label_map_path(const std::string& value) {
  _has_bits_[0] |= 0x00000001u;
  label_map_path_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArena());
}
inline std::string* DetectionLabelIdToTextCalculatorOptions::_internal_mutable_label_map_path() {
  _has_bits_[0] |= 0x00000001u;
  return label_map_path_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArena());
}
inline std::string* DetectionLabelIdToTextCalculatorOptions::release_label_map_path() {
  // @@protoc_insertion_point(field_release:mediapipe.DetectionLabelIdToTextCalculatorOptions.label_map_path)
  if (!_internal_has_label_map_path()) {
    return nullptr;
  }
  _has_bits_[0] &= ~0x00000001u;
  return label_map_path_.ReleaseNonDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
}
inline void DetectionLabelIdToTextCalculatorOptions::set_allocated_label_map_path(std::string* label_map_path) {
  if (label_map_path != nullptr) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  label_map_path_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), label_map_path,
      GetArena());
  // @@protoc_insertion_point(field_set_allocated:mediapipe.DetectionLabelIdToTextCalculatorOptions.label_map_path)
}

// repeated string label = 2;
inline int DetectionLabelIdToTextCalculatorOptions::_internal_label_size() const {
  return label_.size();
}
inline int DetectionLabelIdToTextCalculatorOptions::label_size() const {
  return _internal_label_size();
}
inline void DetectionLabelIdToTextCalculatorOptions::clear_label() {
  label_.Clear();
}
inline std::string* DetectionLabelIdToTextCalculatorOptions::add_label() {
  // @@protoc_insertion_point(field_add_mutable:mediapipe.DetectionLabelIdToTextCalculatorOptions.label)
  return _internal_add_label();
}
inline const std::string& DetectionLabelIdToTextCalculatorOptions::_internal_label(int index) const {
  return label_.Get(index);
}
inline const std::string& DetectionLabelIdToTextCalculatorOptions::label(int index) const {
  // @@protoc_insertion_point(field_get:mediapipe.DetectionLabelIdToTextCalculatorOptions.label)
  return _internal_label(index);
}
inline std::string* DetectionLabelIdToTextCalculatorOptions::mutable_label(int index) {
  // @@protoc_insertion_point(field_mutable:mediapipe.DetectionLabelIdToTextCalculatorOptions.label)
  return label_.Mutable(index);
}
inline void DetectionLabelIdToTextCalculatorOptions::set_label(int index, const std::string& value) {
  // @@protoc_insertion_point(field_set:mediapipe.DetectionLabelIdToTextCalculatorOptions.label)
  label_.Mutable(index)->assign(value);
}
inline void DetectionLabelIdToTextCalculatorOptions::set_label(int index, std::string&& value) {
  // @@protoc_insertion_point(field_set:mediapipe.DetectionLabelIdToTextCalculatorOptions.label)
  label_.Mutable(index)->assign(std::move(value));
}
inline void DetectionLabelIdToTextCalculatorOptions::set_label(int index, const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  label_.Mutable(index)->assign(value);
  // @@protoc_insertion_point(field_set_char:mediapipe.DetectionLabelIdToTextCalculatorOptions.label)
}
inline void DetectionLabelIdToTextCalculatorOptions::set_label(int index, const char* value, size_t size) {
  label_.Mutable(index)->assign(
    reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:mediapipe.DetectionLabelIdToTextCalculatorOptions.label)
}
inline std::string* DetectionLabelIdToTextCalculatorOptions::_internal_add_label() {
  return label_.Add();
}
inline void DetectionLabelIdToTextCalculatorOptions::add_label(const std::string& value) {
  label_.Add()->assign(value);
  // @@protoc_insertion_point(field_add:mediapipe.DetectionLabelIdToTextCalculatorOptions.label)
}
inline void DetectionLabelIdToTextCalculatorOptions::add_label(std::string&& value) {
  label_.Add(std::move(value));
  // @@protoc_insertion_point(field_add:mediapipe.DetectionLabelIdToTextCalculatorOptions.label)
}
inline void DetectionLabelIdToTextCalculatorOptions::add_label(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  label_.Add()->assign(value);
  // @@protoc_insertion_point(field_add_char:mediapipe.DetectionLabelIdToTextCalculatorOptions.label)
}
inline void DetectionLabelIdToTextCalculatorOptions::add_label(const char* value, size_t size) {
  label_.Add()->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_add_pointer:mediapipe.DetectionLabelIdToTextCalculatorOptions.label)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>&
DetectionLabelIdToTextCalculatorOptions::label() const {
  // @@protoc_insertion_point(field_list:mediapipe.DetectionLabelIdToTextCalculatorOptions.label)
  return label_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>*
DetectionLabelIdToTextCalculatorOptions::mutable_label() {
  // @@protoc_insertion_point(field_mutable_list:mediapipe.DetectionLabelIdToTextCalculatorOptions.label)
  return &label_;
}

// optional bool keep_label_id = 3;
inline bool DetectionLabelIdToTextCalculatorOptions::_internal_has_keep_label_id() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool DetectionLabelIdToTextCalculatorOptions::has_keep_label_id() const {
  return _internal_has_keep_label_id();
}
inline void DetectionLabelIdToTextCalculatorOptions::clear_keep_label_id() {
  keep_label_id_ = false;
  _has_bits_[0] &= ~0x00000002u;
}
inline bool DetectionLabelIdToTextCalculatorOptions::_internal_keep_label_id() const {
  return keep_label_id_;
}
inline bool DetectionLabelIdToTextCalculatorOptions::keep_label_id() const {
  // @@protoc_insertion_point(field_get:mediapipe.DetectionLabelIdToTextCalculatorOptions.keep_label_id)
  return _internal_keep_label_id();
}
inline void DetectionLabelIdToTextCalculatorOptions::_internal_set_keep_label_id(bool value) {
  _has_bits_[0] |= 0x00000002u;
  keep_label_id_ = value;
}
inline void DetectionLabelIdToTextCalculatorOptions::set_keep_label_id(bool value) {
  _internal_set_keep_label_id(value);
  // @@protoc_insertion_point(field_set:mediapipe.DetectionLabelIdToTextCalculatorOptions.keep_label_id)
}

// map<int64, .mediapipe.LabelMapItem> label_items = 4;
inline int DetectionLabelIdToTextCalculatorOptions::_internal_label_items_size() const {
  return label_items_.size();
}
inline int DetectionLabelIdToTextCalculatorOptions::label_items_size() const {
  return _internal_label_items_size();
}
inline const ::PROTOBUF_NAMESPACE_ID::Map< ::PROTOBUF_NAMESPACE_ID::int64, ::mediapipe::LabelMapItem >&
DetectionLabelIdToTextCalculatorOptions::_internal_label_items() const {
  return label_items_.GetMap();
}
inline const ::PROTOBUF_NAMESPACE_ID::Map< ::PROTOBUF_NAMESPACE_ID::int64, ::mediapipe::LabelMapItem >&
DetectionLabelIdToTextCalculatorOptions::label_items() const {
  // @@protoc_insertion_point(field_map:mediapipe.DetectionLabelIdToTextCalculatorOptions.label_items)
  return _internal_label_items();
}
inline ::PROTOBUF_NAMESPACE_ID::Map< ::PROTOBUF_NAMESPACE_ID::int64, ::mediapipe::LabelMapItem >*
DetectionLabelIdToTextCalculatorOptions::_internal_mutable_label_items() {
  return label_items_.MutableMap();
}
inline ::PROTOBUF_NAMESPACE_ID::Map< ::PROTOBUF_NAMESPACE_ID::int64, ::mediapipe::LabelMapItem >*
DetectionLabelIdToTextCalculatorOptions::mutable_label_items() {
  // @@protoc_insertion_point(field_mutable_map:mediapipe.DetectionLabelIdToTextCalculatorOptions.label_items)
  return _internal_mutable_label_items();
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <x/google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto
