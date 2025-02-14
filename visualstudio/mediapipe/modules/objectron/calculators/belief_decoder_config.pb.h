// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/modules/objectron/calculators/belief_decoder_config.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto

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
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto;
namespace mediapipe {
class BeliefDecoderConfig;
struct BeliefDecoderConfigDefaultTypeInternal;
extern BeliefDecoderConfigDefaultTypeInternal _BeliefDecoderConfig_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::BeliefDecoderConfig* Arena::CreateMaybeMessage<::mediapipe::BeliefDecoderConfig>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class BeliefDecoderConfig PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.BeliefDecoderConfig) */ {
 public:
  inline BeliefDecoderConfig() : BeliefDecoderConfig(nullptr) {}
  ~BeliefDecoderConfig() override;
  explicit constexpr BeliefDecoderConfig(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  BeliefDecoderConfig(const BeliefDecoderConfig& from);
  BeliefDecoderConfig(BeliefDecoderConfig&& from) noexcept
    : BeliefDecoderConfig() {
    *this = ::std::move(from);
  }

  inline BeliefDecoderConfig& operator=(const BeliefDecoderConfig& from) {
    CopyFrom(from);
    return *this;
  }
  inline BeliefDecoderConfig& operator=(BeliefDecoderConfig&& from) noexcept {
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
  static const BeliefDecoderConfig& default_instance() {
    return *internal_default_instance();
  }
  static inline const BeliefDecoderConfig* internal_default_instance() {
    return reinterpret_cast<const BeliefDecoderConfig*>(
               &_BeliefDecoderConfig_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(BeliefDecoderConfig& a, BeliefDecoderConfig& b) {
    a.Swap(&b);
  }
  inline void Swap(BeliefDecoderConfig* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(BeliefDecoderConfig* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline BeliefDecoderConfig* New() const final {
    return CreateMaybeMessage<BeliefDecoderConfig>(nullptr);
  }

  BeliefDecoderConfig* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<BeliefDecoderConfig>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const BeliefDecoderConfig& from);
  void MergeFrom(const BeliefDecoderConfig& from);
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
  void InternalSwap(BeliefDecoderConfig* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.BeliefDecoderConfig";
  }
  protected:
  explicit BeliefDecoderConfig(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kVotingRadiusFieldNumber = 4,
    kVotingAllowanceFieldNumber = 5,
    kVotingThresholdFieldNumber = 6,
    kOffsetScaleCoefFieldNumber = 3,
    kHeatmapThresholdFieldNumber = 1,
    kLocalMaxDistanceFieldNumber = 2,
  };
  // optional int32 voting_radius = 4;
  bool has_voting_radius() const;
  private:
  bool _internal_has_voting_radius() const;
  public:
  void clear_voting_radius();
  ::PROTOBUF_NAMESPACE_ID::int32 voting_radius() const;
  void set_voting_radius(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_voting_radius() const;
  void _internal_set_voting_radius(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional int32 voting_allowance = 5;
  bool has_voting_allowance() const;
  private:
  bool _internal_has_voting_allowance() const;
  public:
  void clear_voting_allowance();
  ::PROTOBUF_NAMESPACE_ID::int32 voting_allowance() const;
  void set_voting_allowance(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_voting_allowance() const;
  void _internal_set_voting_allowance(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional float voting_threshold = 6;
  bool has_voting_threshold() const;
  private:
  bool _internal_has_voting_threshold() const;
  public:
  void clear_voting_threshold();
  float voting_threshold() const;
  void set_voting_threshold(float value);
  private:
  float _internal_voting_threshold() const;
  void _internal_set_voting_threshold(float value);
  public:

  // optional float offset_scale_coef = 3 [default = 0.5, deprecated = true];
  PROTOBUF_DEPRECATED bool has_offset_scale_coef() const;
  private:
  bool _internal_has_offset_scale_coef() const;
  public:
  PROTOBUF_DEPRECATED void clear_offset_scale_coef();
  PROTOBUF_DEPRECATED float offset_scale_coef() const;
  PROTOBUF_DEPRECATED void set_offset_scale_coef(float value);
  private:
  float _internal_offset_scale_coef() const;
  void _internal_set_offset_scale_coef(float value);
  public:

  // optional float heatmap_threshold = 1 [default = 0.9];
  bool has_heatmap_threshold() const;
  private:
  bool _internal_has_heatmap_threshold() const;
  public:
  void clear_heatmap_threshold();
  float heatmap_threshold() const;
  void set_heatmap_threshold(float value);
  private:
  float _internal_heatmap_threshold() const;
  void _internal_set_heatmap_threshold(float value);
  public:

  // optional float local_max_distance = 2 [default = 10];
  bool has_local_max_distance() const;
  private:
  bool _internal_has_local_max_distance() const;
  public:
  void clear_local_max_distance();
  float local_max_distance() const;
  void set_local_max_distance(float value);
  private:
  float _internal_local_max_distance() const;
  void _internal_set_local_max_distance(float value);
  public:

  // @@protoc_insertion_point(class_scope:mediapipe.BeliefDecoderConfig)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::int32 voting_radius_;
  ::PROTOBUF_NAMESPACE_ID::int32 voting_allowance_;
  float voting_threshold_;
  float offset_scale_coef_;
  float heatmap_threshold_;
  float local_max_distance_;
  friend struct ::TableStruct_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// BeliefDecoderConfig

// optional float heatmap_threshold = 1 [default = 0.9];
inline bool BeliefDecoderConfig::_internal_has_heatmap_threshold() const {
  bool value = (_has_bits_[0] & 0x00000010u) != 0;
  return value;
}
inline bool BeliefDecoderConfig::has_heatmap_threshold() const {
  return _internal_has_heatmap_threshold();
}
inline void BeliefDecoderConfig::clear_heatmap_threshold() {
  heatmap_threshold_ = 0.9f;
  _has_bits_[0] &= ~0x00000010u;
}
inline float BeliefDecoderConfig::_internal_heatmap_threshold() const {
  return heatmap_threshold_;
}
inline float BeliefDecoderConfig::heatmap_threshold() const {
  // @@protoc_insertion_point(field_get:mediapipe.BeliefDecoderConfig.heatmap_threshold)
  return _internal_heatmap_threshold();
}
inline void BeliefDecoderConfig::_internal_set_heatmap_threshold(float value) {
  _has_bits_[0] |= 0x00000010u;
  heatmap_threshold_ = value;
}
inline void BeliefDecoderConfig::set_heatmap_threshold(float value) {
  _internal_set_heatmap_threshold(value);
  // @@protoc_insertion_point(field_set:mediapipe.BeliefDecoderConfig.heatmap_threshold)
}

// optional float local_max_distance = 2 [default = 10];
inline bool BeliefDecoderConfig::_internal_has_local_max_distance() const {
  bool value = (_has_bits_[0] & 0x00000020u) != 0;
  return value;
}
inline bool BeliefDecoderConfig::has_local_max_distance() const {
  return _internal_has_local_max_distance();
}
inline void BeliefDecoderConfig::clear_local_max_distance() {
  local_max_distance_ = 10;
  _has_bits_[0] &= ~0x00000020u;
}
inline float BeliefDecoderConfig::_internal_local_max_distance() const {
  return local_max_distance_;
}
inline float BeliefDecoderConfig::local_max_distance() const {
  // @@protoc_insertion_point(field_get:mediapipe.BeliefDecoderConfig.local_max_distance)
  return _internal_local_max_distance();
}
inline void BeliefDecoderConfig::_internal_set_local_max_distance(float value) {
  _has_bits_[0] |= 0x00000020u;
  local_max_distance_ = value;
}
inline void BeliefDecoderConfig::set_local_max_distance(float value) {
  _internal_set_local_max_distance(value);
  // @@protoc_insertion_point(field_set:mediapipe.BeliefDecoderConfig.local_max_distance)
}

// optional float offset_scale_coef = 3 [default = 0.5, deprecated = true];
inline bool BeliefDecoderConfig::_internal_has_offset_scale_coef() const {
  bool value = (_has_bits_[0] & 0x00000008u) != 0;
  return value;
}
inline bool BeliefDecoderConfig::has_offset_scale_coef() const {
  return _internal_has_offset_scale_coef();
}
inline void BeliefDecoderConfig::clear_offset_scale_coef() {
  offset_scale_coef_ = 0.5f;
  _has_bits_[0] &= ~0x00000008u;
}
inline float BeliefDecoderConfig::_internal_offset_scale_coef() const {
  return offset_scale_coef_;
}
inline float BeliefDecoderConfig::offset_scale_coef() const {
  // @@protoc_insertion_point(field_get:mediapipe.BeliefDecoderConfig.offset_scale_coef)
  return _internal_offset_scale_coef();
}
inline void BeliefDecoderConfig::_internal_set_offset_scale_coef(float value) {
  _has_bits_[0] |= 0x00000008u;
  offset_scale_coef_ = value;
}
inline void BeliefDecoderConfig::set_offset_scale_coef(float value) {
  _internal_set_offset_scale_coef(value);
  // @@protoc_insertion_point(field_set:mediapipe.BeliefDecoderConfig.offset_scale_coef)
}

// optional int32 voting_radius = 4;
inline bool BeliefDecoderConfig::_internal_has_voting_radius() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool BeliefDecoderConfig::has_voting_radius() const {
  return _internal_has_voting_radius();
}
inline void BeliefDecoderConfig::clear_voting_radius() {
  voting_radius_ = 0;
  _has_bits_[0] &= ~0x00000001u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 BeliefDecoderConfig::_internal_voting_radius() const {
  return voting_radius_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 BeliefDecoderConfig::voting_radius() const {
  // @@protoc_insertion_point(field_get:mediapipe.BeliefDecoderConfig.voting_radius)
  return _internal_voting_radius();
}
inline void BeliefDecoderConfig::_internal_set_voting_radius(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000001u;
  voting_radius_ = value;
}
inline void BeliefDecoderConfig::set_voting_radius(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_voting_radius(value);
  // @@protoc_insertion_point(field_set:mediapipe.BeliefDecoderConfig.voting_radius)
}

// optional int32 voting_allowance = 5;
inline bool BeliefDecoderConfig::_internal_has_voting_allowance() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool BeliefDecoderConfig::has_voting_allowance() const {
  return _internal_has_voting_allowance();
}
inline void BeliefDecoderConfig::clear_voting_allowance() {
  voting_allowance_ = 0;
  _has_bits_[0] &= ~0x00000002u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 BeliefDecoderConfig::_internal_voting_allowance() const {
  return voting_allowance_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 BeliefDecoderConfig::voting_allowance() const {
  // @@protoc_insertion_point(field_get:mediapipe.BeliefDecoderConfig.voting_allowance)
  return _internal_voting_allowance();
}
inline void BeliefDecoderConfig::_internal_set_voting_allowance(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000002u;
  voting_allowance_ = value;
}
inline void BeliefDecoderConfig::set_voting_allowance(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_voting_allowance(value);
  // @@protoc_insertion_point(field_set:mediapipe.BeliefDecoderConfig.voting_allowance)
}

// optional float voting_threshold = 6;
inline bool BeliefDecoderConfig::_internal_has_voting_threshold() const {
  bool value = (_has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool BeliefDecoderConfig::has_voting_threshold() const {
  return _internal_has_voting_threshold();
}
inline void BeliefDecoderConfig::clear_voting_threshold() {
  voting_threshold_ = 0;
  _has_bits_[0] &= ~0x00000004u;
}
inline float BeliefDecoderConfig::_internal_voting_threshold() const {
  return voting_threshold_;
}
inline float BeliefDecoderConfig::voting_threshold() const {
  // @@protoc_insertion_point(field_get:mediapipe.BeliefDecoderConfig.voting_threshold)
  return _internal_voting_threshold();
}
inline void BeliefDecoderConfig::_internal_set_voting_threshold(float value) {
  _has_bits_[0] |= 0x00000004u;
  voting_threshold_ = value;
}
inline void BeliefDecoderConfig::set_voting_threshold(float value) {
  _internal_set_voting_threshold(value);
  // @@protoc_insertion_point(field_set:mediapipe.BeliefDecoderConfig.voting_threshold)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto
