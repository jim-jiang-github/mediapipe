// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/video/flow_packager_calculator.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto

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
#include "mediapipe/util/tracking/flow_packager.pb.h"
// @@protoc_insertion_point(includes)
#include <x/google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto;
namespace mediapipe {
class FlowPackagerCalculatorOptions;
struct FlowPackagerCalculatorOptionsDefaultTypeInternal;
extern FlowPackagerCalculatorOptionsDefaultTypeInternal _FlowPackagerCalculatorOptions_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::FlowPackagerCalculatorOptions* Arena::CreateMaybeMessage<::mediapipe::FlowPackagerCalculatorOptions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class FlowPackagerCalculatorOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.FlowPackagerCalculatorOptions) */ {
 public:
  inline FlowPackagerCalculatorOptions() : FlowPackagerCalculatorOptions(nullptr) {}
  ~FlowPackagerCalculatorOptions() override;
  explicit constexpr FlowPackagerCalculatorOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  FlowPackagerCalculatorOptions(const FlowPackagerCalculatorOptions& from);
  FlowPackagerCalculatorOptions(FlowPackagerCalculatorOptions&& from) noexcept
    : FlowPackagerCalculatorOptions() {
    *this = ::std::move(from);
  }

  inline FlowPackagerCalculatorOptions& operator=(const FlowPackagerCalculatorOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline FlowPackagerCalculatorOptions& operator=(FlowPackagerCalculatorOptions&& from) noexcept {
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
  static const FlowPackagerCalculatorOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const FlowPackagerCalculatorOptions* internal_default_instance() {
    return reinterpret_cast<const FlowPackagerCalculatorOptions*>(
               &_FlowPackagerCalculatorOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(FlowPackagerCalculatorOptions& a, FlowPackagerCalculatorOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(FlowPackagerCalculatorOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(FlowPackagerCalculatorOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline FlowPackagerCalculatorOptions* New() const final {
    return CreateMaybeMessage<FlowPackagerCalculatorOptions>(nullptr);
  }

  FlowPackagerCalculatorOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<FlowPackagerCalculatorOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const FlowPackagerCalculatorOptions& from);
  void MergeFrom(const FlowPackagerCalculatorOptions& from);
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
  void InternalSwap(FlowPackagerCalculatorOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.FlowPackagerCalculatorOptions";
  }
  protected:
  explicit FlowPackagerCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kCacheFileFormatFieldNumber = 3,
    kFlowPackagerOptionsFieldNumber = 1,
    kCachingChunkSizeMsecFieldNumber = 2,
  };
  // optional string cache_file_format = 3 [default = "chunk_%04d"];
  bool has_cache_file_format() const;
  private:
  bool _internal_has_cache_file_format() const;
  public:
  void clear_cache_file_format();
  const std::string& cache_file_format() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_cache_file_format(ArgT0&& arg0, ArgT... args);
  std::string* mutable_cache_file_format();
  std::string* release_cache_file_format();
  void set_allocated_cache_file_format(std::string* cache_file_format);
  private:
  const std::string& _internal_cache_file_format() const;
  void _internal_set_cache_file_format(const std::string& value);
  std::string* _internal_mutable_cache_file_format();
  public:

  // optional .mediapipe.FlowPackagerOptions flow_packager_options = 1;
  bool has_flow_packager_options() const;
  private:
  bool _internal_has_flow_packager_options() const;
  public:
  void clear_flow_packager_options();
  const ::mediapipe::FlowPackagerOptions& flow_packager_options() const;
  PROTOBUF_FUTURE_MUST_USE_RESULT ::mediapipe::FlowPackagerOptions* release_flow_packager_options();
  ::mediapipe::FlowPackagerOptions* mutable_flow_packager_options();
  void set_allocated_flow_packager_options(::mediapipe::FlowPackagerOptions* flow_packager_options);
  private:
  const ::mediapipe::FlowPackagerOptions& _internal_flow_packager_options() const;
  ::mediapipe::FlowPackagerOptions* _internal_mutable_flow_packager_options();
  public:
  void unsafe_arena_set_allocated_flow_packager_options(
      ::mediapipe::FlowPackagerOptions* flow_packager_options);
  ::mediapipe::FlowPackagerOptions* unsafe_arena_release_flow_packager_options();

  // optional int32 caching_chunk_size_msec = 2 [default = 2500];
  bool has_caching_chunk_size_msec() const;
  private:
  bool _internal_has_caching_chunk_size_msec() const;
  public:
  void clear_caching_chunk_size_msec();
  ::PROTOBUF_NAMESPACE_ID::int32 caching_chunk_size_msec() const;
  void set_caching_chunk_size_msec(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_caching_chunk_size_msec() const;
  void _internal_set_caching_chunk_size_msec(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  static const int kExtFieldNumber = 271236147;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::FlowPackagerCalculatorOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.FlowPackagerCalculatorOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  static const ::PROTOBUF_NAMESPACE_ID::internal::LazyString _i_give_permission_to_break_this_code_default_cache_file_format_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr cache_file_format_;
  ::mediapipe::FlowPackagerOptions* flow_packager_options_;
  ::PROTOBUF_NAMESPACE_ID::int32 caching_chunk_size_msec_;
  friend struct ::TableStruct_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// FlowPackagerCalculatorOptions

// optional .mediapipe.FlowPackagerOptions flow_packager_options = 1;
inline bool FlowPackagerCalculatorOptions::_internal_has_flow_packager_options() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  PROTOBUF_ASSUME(!value || flow_packager_options_ != nullptr);
  return value;
}
inline bool FlowPackagerCalculatorOptions::has_flow_packager_options() const {
  return _internal_has_flow_packager_options();
}
inline const ::mediapipe::FlowPackagerOptions& FlowPackagerCalculatorOptions::_internal_flow_packager_options() const {
  const ::mediapipe::FlowPackagerOptions* p = flow_packager_options_;
  return p != nullptr ? *p : reinterpret_cast<const ::mediapipe::FlowPackagerOptions&>(
      ::mediapipe::_FlowPackagerOptions_default_instance_);
}
inline const ::mediapipe::FlowPackagerOptions& FlowPackagerCalculatorOptions::flow_packager_options() const {
  // @@protoc_insertion_point(field_get:mediapipe.FlowPackagerCalculatorOptions.flow_packager_options)
  return _internal_flow_packager_options();
}
inline void FlowPackagerCalculatorOptions::unsafe_arena_set_allocated_flow_packager_options(
    ::mediapipe::FlowPackagerOptions* flow_packager_options) {
  if (GetArena() == nullptr) {
    delete reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(flow_packager_options_);
  }
  flow_packager_options_ = flow_packager_options;
  if (flow_packager_options) {
    _has_bits_[0] |= 0x00000002u;
  } else {
    _has_bits_[0] &= ~0x00000002u;
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:mediapipe.FlowPackagerCalculatorOptions.flow_packager_options)
}
inline ::mediapipe::FlowPackagerOptions* FlowPackagerCalculatorOptions::release_flow_packager_options() {
  _has_bits_[0] &= ~0x00000002u;
  ::mediapipe::FlowPackagerOptions* temp = flow_packager_options_;
  flow_packager_options_ = nullptr;
  if (GetArena() != nullptr) {
    temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  }
  return temp;
}
inline ::mediapipe::FlowPackagerOptions* FlowPackagerCalculatorOptions::unsafe_arena_release_flow_packager_options() {
  // @@protoc_insertion_point(field_release:mediapipe.FlowPackagerCalculatorOptions.flow_packager_options)
  _has_bits_[0] &= ~0x00000002u;
  ::mediapipe::FlowPackagerOptions* temp = flow_packager_options_;
  flow_packager_options_ = nullptr;
  return temp;
}
inline ::mediapipe::FlowPackagerOptions* FlowPackagerCalculatorOptions::_internal_mutable_flow_packager_options() {
  _has_bits_[0] |= 0x00000002u;
  if (flow_packager_options_ == nullptr) {
    auto* p = CreateMaybeMessage<::mediapipe::FlowPackagerOptions>(GetArena());
    flow_packager_options_ = p;
  }
  return flow_packager_options_;
}
inline ::mediapipe::FlowPackagerOptions* FlowPackagerCalculatorOptions::mutable_flow_packager_options() {
  // @@protoc_insertion_point(field_mutable:mediapipe.FlowPackagerCalculatorOptions.flow_packager_options)
  return _internal_mutable_flow_packager_options();
}
inline void FlowPackagerCalculatorOptions::set_allocated_flow_packager_options(::mediapipe::FlowPackagerOptions* flow_packager_options) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArena();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(flow_packager_options_);
  }
  if (flow_packager_options) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
      reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(flow_packager_options)->GetArena();
    if (message_arena != submessage_arena) {
      flow_packager_options = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, flow_packager_options, submessage_arena);
    }
    _has_bits_[0] |= 0x00000002u;
  } else {
    _has_bits_[0] &= ~0x00000002u;
  }
  flow_packager_options_ = flow_packager_options;
  // @@protoc_insertion_point(field_set_allocated:mediapipe.FlowPackagerCalculatorOptions.flow_packager_options)
}

// optional int32 caching_chunk_size_msec = 2 [default = 2500];
inline bool FlowPackagerCalculatorOptions::_internal_has_caching_chunk_size_msec() const {
  bool value = (_has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool FlowPackagerCalculatorOptions::has_caching_chunk_size_msec() const {
  return _internal_has_caching_chunk_size_msec();
}
inline void FlowPackagerCalculatorOptions::clear_caching_chunk_size_msec() {
  caching_chunk_size_msec_ = 2500;
  _has_bits_[0] &= ~0x00000004u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 FlowPackagerCalculatorOptions::_internal_caching_chunk_size_msec() const {
  return caching_chunk_size_msec_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 FlowPackagerCalculatorOptions::caching_chunk_size_msec() const {
  // @@protoc_insertion_point(field_get:mediapipe.FlowPackagerCalculatorOptions.caching_chunk_size_msec)
  return _internal_caching_chunk_size_msec();
}
inline void FlowPackagerCalculatorOptions::_internal_set_caching_chunk_size_msec(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000004u;
  caching_chunk_size_msec_ = value;
}
inline void FlowPackagerCalculatorOptions::set_caching_chunk_size_msec(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_caching_chunk_size_msec(value);
  // @@protoc_insertion_point(field_set:mediapipe.FlowPackagerCalculatorOptions.caching_chunk_size_msec)
}

// optional string cache_file_format = 3 [default = "chunk_%04d"];
inline bool FlowPackagerCalculatorOptions::_internal_has_cache_file_format() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool FlowPackagerCalculatorOptions::has_cache_file_format() const {
  return _internal_has_cache_file_format();
}
inline void FlowPackagerCalculatorOptions::clear_cache_file_format() {
  cache_file_format_.ClearToDefault(::mediapipe::FlowPackagerCalculatorOptions::_i_give_permission_to_break_this_code_default_cache_file_format_, GetArena());
  _has_bits_[0] &= ~0x00000001u;
}
inline const std::string& FlowPackagerCalculatorOptions::cache_file_format() const {
  // @@protoc_insertion_point(field_get:mediapipe.FlowPackagerCalculatorOptions.cache_file_format)
  if (cache_file_format_.IsDefault(nullptr)) return _i_give_permission_to_break_this_code_default_cache_file_format_.get();
  return _internal_cache_file_format();
}
template <typename ArgT0, typename... ArgT>
PROTOBUF_ALWAYS_INLINE
inline void FlowPackagerCalculatorOptions::set_cache_file_format(ArgT0&& arg0, ArgT... args) {
 _has_bits_[0] |= 0x00000001u;
 cache_file_format_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::NonEmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArena());
  // @@protoc_insertion_point(field_set:mediapipe.FlowPackagerCalculatorOptions.cache_file_format)
}
inline std::string* FlowPackagerCalculatorOptions::mutable_cache_file_format() {
  // @@protoc_insertion_point(field_mutable:mediapipe.FlowPackagerCalculatorOptions.cache_file_format)
  return _internal_mutable_cache_file_format();
}
inline const std::string& FlowPackagerCalculatorOptions::_internal_cache_file_format() const {
  return cache_file_format_.Get();
}
inline void FlowPackagerCalculatorOptions::_internal_set_cache_file_format(const std::string& value) {
  _has_bits_[0] |= 0x00000001u;
  cache_file_format_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::NonEmptyDefault{}, value, GetArena());
}
inline std::string* FlowPackagerCalculatorOptions::_internal_mutable_cache_file_format() {
  _has_bits_[0] |= 0x00000001u;
  return cache_file_format_.Mutable(::mediapipe::FlowPackagerCalculatorOptions::_i_give_permission_to_break_this_code_default_cache_file_format_, GetArena());
}
inline std::string* FlowPackagerCalculatorOptions::release_cache_file_format() {
  // @@protoc_insertion_point(field_release:mediapipe.FlowPackagerCalculatorOptions.cache_file_format)
  if (!_internal_has_cache_file_format()) {
    return nullptr;
  }
  _has_bits_[0] &= ~0x00000001u;
  return cache_file_format_.ReleaseNonDefault(nullptr, GetArena());
}
inline void FlowPackagerCalculatorOptions::set_allocated_cache_file_format(std::string* cache_file_format) {
  if (cache_file_format != nullptr) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  cache_file_format_.SetAllocated(nullptr, cache_file_format,
      GetArena());
  // @@protoc_insertion_point(field_set_allocated:mediapipe.FlowPackagerCalculatorOptions.cache_file_format)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <x/google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto
