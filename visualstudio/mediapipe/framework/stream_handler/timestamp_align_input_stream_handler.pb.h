// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/framework/stream_handler/timestamp_align_input_stream_handler.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fframework_2fstream_5fhandler_2ftimestamp_5falign_5finput_5fstream_5fhandler_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fframework_2fstream_5fhandler_2ftimestamp_5falign_5finput_5fstream_5fhandler_2eproto

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
#include "mediapipe/framework/mediapipe_options.pb.h"
// @@protoc_insertion_point(includes)
#include <x/google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fframework_2fstream_5fhandler_2ftimestamp_5falign_5finput_5fstream_5fhandler_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fframework_2fstream_5fhandler_2ftimestamp_5falign_5finput_5fstream_5fhandler_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fframework_2fstream_5fhandler_2ftimestamp_5falign_5finput_5fstream_5fhandler_2eproto;
namespace mediapipe {
class TimestampAlignInputStreamHandlerOptions;
struct TimestampAlignInputStreamHandlerOptionsDefaultTypeInternal;
extern TimestampAlignInputStreamHandlerOptionsDefaultTypeInternal _TimestampAlignInputStreamHandlerOptions_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::TimestampAlignInputStreamHandlerOptions* Arena::CreateMaybeMessage<::mediapipe::TimestampAlignInputStreamHandlerOptions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class TimestampAlignInputStreamHandlerOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.TimestampAlignInputStreamHandlerOptions) */ {
 public:
  inline TimestampAlignInputStreamHandlerOptions() : TimestampAlignInputStreamHandlerOptions(nullptr) {}
  ~TimestampAlignInputStreamHandlerOptions() override;
  explicit constexpr TimestampAlignInputStreamHandlerOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  TimestampAlignInputStreamHandlerOptions(const TimestampAlignInputStreamHandlerOptions& from);
  TimestampAlignInputStreamHandlerOptions(TimestampAlignInputStreamHandlerOptions&& from) noexcept
    : TimestampAlignInputStreamHandlerOptions() {
    *this = ::std::move(from);
  }

  inline TimestampAlignInputStreamHandlerOptions& operator=(const TimestampAlignInputStreamHandlerOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline TimestampAlignInputStreamHandlerOptions& operator=(TimestampAlignInputStreamHandlerOptions&& from) noexcept {
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
  static const TimestampAlignInputStreamHandlerOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const TimestampAlignInputStreamHandlerOptions* internal_default_instance() {
    return reinterpret_cast<const TimestampAlignInputStreamHandlerOptions*>(
               &_TimestampAlignInputStreamHandlerOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(TimestampAlignInputStreamHandlerOptions& a, TimestampAlignInputStreamHandlerOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(TimestampAlignInputStreamHandlerOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(TimestampAlignInputStreamHandlerOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline TimestampAlignInputStreamHandlerOptions* New() const final {
    return CreateMaybeMessage<TimestampAlignInputStreamHandlerOptions>(nullptr);
  }

  TimestampAlignInputStreamHandlerOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<TimestampAlignInputStreamHandlerOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const TimestampAlignInputStreamHandlerOptions& from);
  void MergeFrom(const TimestampAlignInputStreamHandlerOptions& from);
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
  void InternalSwap(TimestampAlignInputStreamHandlerOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.TimestampAlignInputStreamHandlerOptions";
  }
  protected:
  explicit TimestampAlignInputStreamHandlerOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kTimestampBaseTagIndexFieldNumber = 1,
  };
  // optional string timestamp_base_tag_index = 1;
  bool has_timestamp_base_tag_index() const;
  private:
  bool _internal_has_timestamp_base_tag_index() const;
  public:
  void clear_timestamp_base_tag_index();
  const std::string& timestamp_base_tag_index() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_timestamp_base_tag_index(ArgT0&& arg0, ArgT... args);
  std::string* mutable_timestamp_base_tag_index();
  std::string* release_timestamp_base_tag_index();
  void set_allocated_timestamp_base_tag_index(std::string* timestamp_base_tag_index);
  private:
  const std::string& _internal_timestamp_base_tag_index() const;
  void _internal_set_timestamp_base_tag_index(const std::string& value);
  std::string* _internal_mutable_timestamp_base_tag_index();
  public:

  static const int kExtFieldNumber = 190104979;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::MediaPipeOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::TimestampAlignInputStreamHandlerOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.TimestampAlignInputStreamHandlerOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr timestamp_base_tag_index_;
  friend struct ::TableStruct_mediapipe_2fframework_2fstream_5fhandler_2ftimestamp_5falign_5finput_5fstream_5fhandler_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// TimestampAlignInputStreamHandlerOptions

// optional string timestamp_base_tag_index = 1;
inline bool TimestampAlignInputStreamHandlerOptions::_internal_has_timestamp_base_tag_index() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool TimestampAlignInputStreamHandlerOptions::has_timestamp_base_tag_index() const {
  return _internal_has_timestamp_base_tag_index();
}
inline void TimestampAlignInputStreamHandlerOptions::clear_timestamp_base_tag_index() {
  timestamp_base_tag_index_.ClearToEmpty();
  _has_bits_[0] &= ~0x00000001u;
}
inline const std::string& TimestampAlignInputStreamHandlerOptions::timestamp_base_tag_index() const {
  // @@protoc_insertion_point(field_get:mediapipe.TimestampAlignInputStreamHandlerOptions.timestamp_base_tag_index)
  return _internal_timestamp_base_tag_index();
}
template <typename ArgT0, typename... ArgT>
PROTOBUF_ALWAYS_INLINE
inline void TimestampAlignInputStreamHandlerOptions::set_timestamp_base_tag_index(ArgT0&& arg0, ArgT... args) {
 _has_bits_[0] |= 0x00000001u;
 timestamp_base_tag_index_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArena());
  // @@protoc_insertion_point(field_set:mediapipe.TimestampAlignInputStreamHandlerOptions.timestamp_base_tag_index)
}
inline std::string* TimestampAlignInputStreamHandlerOptions::mutable_timestamp_base_tag_index() {
  // @@protoc_insertion_point(field_mutable:mediapipe.TimestampAlignInputStreamHandlerOptions.timestamp_base_tag_index)
  return _internal_mutable_timestamp_base_tag_index();
}
inline const std::string& TimestampAlignInputStreamHandlerOptions::_internal_timestamp_base_tag_index() const {
  return timestamp_base_tag_index_.Get();
}
inline void TimestampAlignInputStreamHandlerOptions::_internal_set_timestamp_base_tag_index(const std::string& value) {
  _has_bits_[0] |= 0x00000001u;
  timestamp_base_tag_index_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArena());
}
inline std::string* TimestampAlignInputStreamHandlerOptions::_internal_mutable_timestamp_base_tag_index() {
  _has_bits_[0] |= 0x00000001u;
  return timestamp_base_tag_index_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArena());
}
inline std::string* TimestampAlignInputStreamHandlerOptions::release_timestamp_base_tag_index() {
  // @@protoc_insertion_point(field_release:mediapipe.TimestampAlignInputStreamHandlerOptions.timestamp_base_tag_index)
  if (!_internal_has_timestamp_base_tag_index()) {
    return nullptr;
  }
  _has_bits_[0] &= ~0x00000001u;
  return timestamp_base_tag_index_.ReleaseNonDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
}
inline void TimestampAlignInputStreamHandlerOptions::set_allocated_timestamp_base_tag_index(std::string* timestamp_base_tag_index) {
  if (timestamp_base_tag_index != nullptr) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  timestamp_base_tag_index_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), timestamp_base_tag_index,
      GetArena());
  // @@protoc_insertion_point(field_set_allocated:mediapipe.TimestampAlignInputStreamHandlerOptions.timestamp_base_tag_index)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <x/google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fframework_2fstream_5fhandler_2ftimestamp_5falign_5finput_5fstream_5fhandler_2eproto
