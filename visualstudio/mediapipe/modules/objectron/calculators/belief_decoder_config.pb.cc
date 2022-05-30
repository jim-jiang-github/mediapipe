// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/modules/objectron/calculators/belief_decoder_config.proto

#include "mediapipe/modules/objectron/calculators/belief_decoder_config.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG
namespace mediapipe {
constexpr BeliefDecoderConfig::BeliefDecoderConfig(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : voting_radius_(0)
  , voting_allowance_(0)
  , voting_threshold_(0)
  , offset_scale_coef_(0.5f)
  , heatmap_threshold_(0.9f)
  , local_max_distance_(10){}
struct BeliefDecoderConfigDefaultTypeInternal {
  constexpr BeliefDecoderConfigDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~BeliefDecoderConfigDefaultTypeInternal() {}
  union {
    BeliefDecoderConfig _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT BeliefDecoderConfigDefaultTypeInternal _BeliefDecoderConfig_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::BeliefDecoderConfig, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::BeliefDecoderConfig, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::BeliefDecoderConfig, heatmap_threshold_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::BeliefDecoderConfig, local_max_distance_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::BeliefDecoderConfig, offset_scale_coef_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::BeliefDecoderConfig, voting_radius_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::BeliefDecoderConfig, voting_allowance_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::BeliefDecoderConfig, voting_threshold_),
  4,
  5,
  3,
  0,
  1,
  2,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 11, sizeof(::mediapipe::BeliefDecoderConfig)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_BeliefDecoderConfig_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\nCmediapipe/modules/objectron/calculator"
  "s/belief_decoder_config.proto\022\tmediapipe"
  "\"\304\001\n\023BeliefDecoderConfig\022\036\n\021heatmap_thre"
  "shold\030\001 \001(\002:\0030.9\022\036\n\022local_max_distance\030\002"
  " \001(\002:\00210\022\"\n\021offset_scale_coef\030\003 \001(\002:\0030.5"
  "B\002\030\001\022\025\n\rvoting_radius\030\004 \001(\005\022\030\n\020voting_al"
  "lowance\030\005 \001(\005\022\030\n\020voting_threshold\030\006 \001(\002"
  ;
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto = {
  false, false, 279, descriptor_table_protodef_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto, "mediapipe/modules/objectron/calculators/belief_decoder_config.proto", 
  &descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto_once, nullptr, 0, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto::offsets,
  file_level_metadata_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto, file_level_enum_descriptors_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto, file_level_service_descriptors_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto_getter() {
  return &descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto(&descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto);
namespace mediapipe {

// ===================================================================

class BeliefDecoderConfig::_Internal {
 public:
  using HasBits = decltype(std::declval<BeliefDecoderConfig>()._has_bits_);
  static void set_has_heatmap_threshold(HasBits* has_bits) {
    (*has_bits)[0] |= 16u;
  }
  static void set_has_local_max_distance(HasBits* has_bits) {
    (*has_bits)[0] |= 32u;
  }
  static void set_has_offset_scale_coef(HasBits* has_bits) {
    (*has_bits)[0] |= 8u;
  }
  static void set_has_voting_radius(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_voting_allowance(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static void set_has_voting_threshold(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
};

BeliefDecoderConfig::BeliefDecoderConfig(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.BeliefDecoderConfig)
}
BeliefDecoderConfig::BeliefDecoderConfig(const BeliefDecoderConfig& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&voting_radius_, &from.voting_radius_,
    static_cast<size_t>(reinterpret_cast<char*>(&local_max_distance_) -
    reinterpret_cast<char*>(&voting_radius_)) + sizeof(local_max_distance_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.BeliefDecoderConfig)
}

void BeliefDecoderConfig::SharedCtor() {
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&voting_radius_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&voting_threshold_) -
    reinterpret_cast<char*>(&voting_radius_)) + sizeof(voting_threshold_));
offset_scale_coef_ = 0.5f;
heatmap_threshold_ = 0.9f;
local_max_distance_ = 10;
}

BeliefDecoderConfig::~BeliefDecoderConfig() {
  // @@protoc_insertion_point(destructor:mediapipe.BeliefDecoderConfig)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void BeliefDecoderConfig::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void BeliefDecoderConfig::ArenaDtor(void* object) {
  BeliefDecoderConfig* _this = reinterpret_cast< BeliefDecoderConfig* >(object);
  (void)_this;
}
void BeliefDecoderConfig::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void BeliefDecoderConfig::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void BeliefDecoderConfig::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.BeliefDecoderConfig)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x0000003fu) {
    ::memset(&voting_radius_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&voting_threshold_) -
        reinterpret_cast<char*>(&voting_radius_)) + sizeof(voting_threshold_));
    offset_scale_coef_ = 0.5f;
    heatmap_threshold_ = 0.9f;
    local_max_distance_ = 10;
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* BeliefDecoderConfig::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional float heatmap_threshold = 1 [default = 0.9];
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 13)) {
          _Internal::set_has_heatmap_threshold(&has_bits);
          heatmap_threshold_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else goto handle_unusual;
        continue;
      // optional float local_max_distance = 2 [default = 10];
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 21)) {
          _Internal::set_has_local_max_distance(&has_bits);
          local_max_distance_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else goto handle_unusual;
        continue;
      // optional float offset_scale_coef = 3 [default = 0.5, deprecated = true];
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 29)) {
          _Internal::set_has_offset_scale_coef(&has_bits);
          offset_scale_coef_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else goto handle_unusual;
        continue;
      // optional int32 voting_radius = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 32)) {
          _Internal::set_has_voting_radius(&has_bits);
          voting_radius_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 voting_allowance = 5;
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 40)) {
          _Internal::set_has_voting_allowance(&has_bits);
          voting_allowance_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional float voting_threshold = 6;
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 53)) {
          _Internal::set_has_voting_threshold(&has_bits);
          voting_threshold_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag == 0) || ((tag & 7) == 4)) {
          CHK_(ptr);
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  _has_bits_.Or(has_bits);
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* BeliefDecoderConfig::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.BeliefDecoderConfig)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional float heatmap_threshold = 1 [default = 0.9];
  if (cached_has_bits & 0x00000010u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(1, this->_internal_heatmap_threshold(), target);
  }

  // optional float local_max_distance = 2 [default = 10];
  if (cached_has_bits & 0x00000020u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(2, this->_internal_local_max_distance(), target);
  }

  // optional float offset_scale_coef = 3 [default = 0.5, deprecated = true];
  if (cached_has_bits & 0x00000008u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(3, this->_internal_offset_scale_coef(), target);
  }

  // optional int32 voting_radius = 4;
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(4, this->_internal_voting_radius(), target);
  }

  // optional int32 voting_allowance = 5;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(5, this->_internal_voting_allowance(), target);
  }

  // optional float voting_threshold = 6;
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(6, this->_internal_voting_threshold(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.BeliefDecoderConfig)
  return target;
}

size_t BeliefDecoderConfig::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.BeliefDecoderConfig)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x0000003fu) {
    // optional int32 voting_radius = 4;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_voting_radius());
    }

    // optional int32 voting_allowance = 5;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_voting_allowance());
    }

    // optional float voting_threshold = 6;
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 + 4;
    }

    // optional float offset_scale_coef = 3 [default = 0.5, deprecated = true];
    if (cached_has_bits & 0x00000008u) {
      total_size += 1 + 4;
    }

    // optional float heatmap_threshold = 1 [default = 0.9];
    if (cached_has_bits & 0x00000010u) {
      total_size += 1 + 4;
    }

    // optional float local_max_distance = 2 [default = 10];
    if (cached_has_bits & 0x00000020u) {
      total_size += 1 + 4;
    }

  }
  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void BeliefDecoderConfig::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.BeliefDecoderConfig)
  GOOGLE_DCHECK_NE(&from, this);
  const BeliefDecoderConfig* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<BeliefDecoderConfig>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.BeliefDecoderConfig)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.BeliefDecoderConfig)
    MergeFrom(*source);
  }
}

void BeliefDecoderConfig::MergeFrom(const BeliefDecoderConfig& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.BeliefDecoderConfig)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x0000003fu) {
    if (cached_has_bits & 0x00000001u) {
      voting_radius_ = from.voting_radius_;
    }
    if (cached_has_bits & 0x00000002u) {
      voting_allowance_ = from.voting_allowance_;
    }
    if (cached_has_bits & 0x00000004u) {
      voting_threshold_ = from.voting_threshold_;
    }
    if (cached_has_bits & 0x00000008u) {
      offset_scale_coef_ = from.offset_scale_coef_;
    }
    if (cached_has_bits & 0x00000010u) {
      heatmap_threshold_ = from.heatmap_threshold_;
    }
    if (cached_has_bits & 0x00000020u) {
      local_max_distance_ = from.local_max_distance_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void BeliefDecoderConfig::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.BeliefDecoderConfig)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void BeliefDecoderConfig::CopyFrom(const BeliefDecoderConfig& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.BeliefDecoderConfig)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool BeliefDecoderConfig::IsInitialized() const {
  return true;
}

void BeliefDecoderConfig::InternalSwap(BeliefDecoderConfig* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(BeliefDecoderConfig, voting_threshold_)
      + sizeof(BeliefDecoderConfig::voting_threshold_)
      - PROTOBUF_FIELD_OFFSET(BeliefDecoderConfig, voting_radius_)>(
          reinterpret_cast<char*>(&voting_radius_),
          reinterpret_cast<char*>(&other->voting_radius_));
  swap(offset_scale_coef_, other->offset_scale_coef_);
  swap(heatmap_threshold_, other->heatmap_threshold_);
  swap(local_max_distance_, other->local_max_distance_);
}

::PROTOBUF_NAMESPACE_ID::Metadata BeliefDecoderConfig::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto_getter, &descriptor_table_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto_once,
      file_level_metadata_mediapipe_2fmodules_2fobjectron_2fcalculators_2fbelief_5fdecoder_5fconfig_2eproto[0]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::BeliefDecoderConfig* Arena::CreateMaybeMessage< ::mediapipe::BeliefDecoderConfig >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::BeliefDecoderConfig >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
