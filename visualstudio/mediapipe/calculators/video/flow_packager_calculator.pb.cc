// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/video/flow_packager_calculator.proto

#include "mediapipe/calculators/video/flow_packager_calculator.pb.h"

#include <algorithm>

#include <x/google/protobuf/io/coded_stream.h>
#include <x/google/protobuf/extension_set.h>
#include <x/google/protobuf/wire_format_lite.h>
#include <x/google/protobuf/descriptor.h>
#include <x/google/protobuf/generated_message_reflection.h>
#include <x/google/protobuf/reflection_ops.h>
#include <x/google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <x/google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG
namespace mediapipe {
constexpr FlowPackagerCalculatorOptions::FlowPackagerCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : cache_file_format_(nullptr)
  , flow_packager_options_(nullptr)
  , caching_chunk_size_msec_(2500){}
struct FlowPackagerCalculatorOptionsDefaultTypeInternal {
  constexpr FlowPackagerCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~FlowPackagerCalculatorOptionsDefaultTypeInternal() {}
  union {
    FlowPackagerCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT FlowPackagerCalculatorOptionsDefaultTypeInternal _FlowPackagerCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::FlowPackagerCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FlowPackagerCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::FlowPackagerCalculatorOptions, flow_packager_options_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FlowPackagerCalculatorOptions, caching_chunk_size_msec_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FlowPackagerCalculatorOptions, cache_file_format_),
  1,
  2,
  0,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 8, sizeof(::mediapipe::FlowPackagerCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_FlowPackagerCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n:mediapipe/calculators/video/flow_packa"
  "ger_calculator.proto\022\tmediapipe\032$mediapi"
  "pe/framework/calculator.proto\032+mediapipe"
  "/util/tracking/flow_packager.proto\"\205\002\n\035F"
  "lowPackagerCalculatorOptions\022=\n\025flow_pac"
  "kager_options\030\001 \001(\0132\036.mediapipe.FlowPack"
  "agerOptions\022%\n\027caching_chunk_size_msec\030\002"
  " \001(\005:\0042500\022%\n\021cache_file_format\030\003 \001(\t:\nc"
  "hunk_%04d2W\n\003ext\022\034.mediapipe.CalculatorO"
  "ptions\030\263\370\252\201\001 \001(\0132(.mediapipe.FlowPackage"
  "rCalculatorOptions"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto_deps[2] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
  &::descriptor_table_mediapipe_2futil_2ftracking_2fflow_5fpackager_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto = {
  false, false, 418, descriptor_table_protodef_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto, "mediapipe/calculators/video/flow_packager_calculator.proto", 
  &descriptor_table_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto_deps, 2, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto(&descriptor_table_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto);
namespace mediapipe {

// ===================================================================

class FlowPackagerCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<FlowPackagerCalculatorOptions>()._has_bits_);
  static const ::mediapipe::FlowPackagerOptions& flow_packager_options(const FlowPackagerCalculatorOptions* msg);
  static void set_has_flow_packager_options(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static void set_has_caching_chunk_size_msec(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static void set_has_cache_file_format(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
};

const ::mediapipe::FlowPackagerOptions&
FlowPackagerCalculatorOptions::_Internal::flow_packager_options(const FlowPackagerCalculatorOptions* msg) {
  return *msg->flow_packager_options_;
}
void FlowPackagerCalculatorOptions::clear_flow_packager_options() {
  if (flow_packager_options_ != nullptr) flow_packager_options_->Clear();
  _has_bits_[0] &= ~0x00000002u;
}
const ::PROTOBUF_NAMESPACE_ID::internal::LazyString FlowPackagerCalculatorOptions::_i_give_permission_to_break_this_code_default_cache_file_format_{{{"chunk_%04d", 10}}, {nullptr}};
FlowPackagerCalculatorOptions::FlowPackagerCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.FlowPackagerCalculatorOptions)
}
FlowPackagerCalculatorOptions::FlowPackagerCalculatorOptions(const FlowPackagerCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  cache_file_format_.UnsafeSetDefault(nullptr);
  if (from._internal_has_cache_file_format()) {
    cache_file_format_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::NonEmptyDefault{}, from._internal_cache_file_format(), 
      GetArena());
  }
  if (from._internal_has_flow_packager_options()) {
    flow_packager_options_ = new ::mediapipe::FlowPackagerOptions(*from.flow_packager_options_);
  } else {
    flow_packager_options_ = nullptr;
  }
  caching_chunk_size_msec_ = from.caching_chunk_size_msec_;
  // @@protoc_insertion_point(copy_constructor:mediapipe.FlowPackagerCalculatorOptions)
}

void FlowPackagerCalculatorOptions::SharedCtor() {
cache_file_format_.UnsafeSetDefault(nullptr);
flow_packager_options_ = nullptr;
caching_chunk_size_msec_ = 2500;
}

FlowPackagerCalculatorOptions::~FlowPackagerCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.FlowPackagerCalculatorOptions)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void FlowPackagerCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
  cache_file_format_.DestroyNoArena(nullptr);
  if (this != internal_default_instance()) delete flow_packager_options_;
}

void FlowPackagerCalculatorOptions::ArenaDtor(void* object) {
  FlowPackagerCalculatorOptions* _this = reinterpret_cast< FlowPackagerCalculatorOptions* >(object);
  (void)_this;
}
void FlowPackagerCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void FlowPackagerCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void FlowPackagerCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.FlowPackagerCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    if (cached_has_bits & 0x00000001u) {
      cache_file_format_.ClearToDefault(::mediapipe::FlowPackagerCalculatorOptions::_i_give_permission_to_break_this_code_default_cache_file_format_, GetArena());
       }
    if (cached_has_bits & 0x00000002u) {
      GOOGLE_DCHECK(flow_packager_options_ != nullptr);
      flow_packager_options_->Clear();
    }
    caching_chunk_size_msec_ = 2500;
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* FlowPackagerCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional .mediapipe.FlowPackagerOptions flow_packager_options = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr = ctx->ParseMessage(_internal_mutable_flow_packager_options(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 caching_chunk_size_msec = 2 [default = 2500];
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          _Internal::set_has_caching_chunk_size_msec(&has_bits);
          caching_chunk_size_msec_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional string cache_file_format = 3 [default = "chunk_%04d"];
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 26)) {
          auto str = _internal_mutable_cache_file_format();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          #ifndef NDEBUG
          ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "mediapipe.FlowPackagerCalculatorOptions.cache_file_format");
          #endif  // !NDEBUG
          CHK_(ptr);
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

::PROTOBUF_NAMESPACE_ID::uint8* FlowPackagerCalculatorOptions::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.FlowPackagerCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional .mediapipe.FlowPackagerOptions flow_packager_options = 1;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        1, _Internal::flow_packager_options(this), target, stream);
  }

  // optional int32 caching_chunk_size_msec = 2 [default = 2500];
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(2, this->_internal_caching_chunk_size_msec(), target);
  }

  // optional string cache_file_format = 3 [default = "chunk_%04d"];
  if (cached_has_bits & 0x00000001u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->_internal_cache_file_format().data(), static_cast<int>(this->_internal_cache_file_format().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "mediapipe.FlowPackagerCalculatorOptions.cache_file_format");
    target = stream->WriteStringMaybeAliased(
        3, this->_internal_cache_file_format(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.FlowPackagerCalculatorOptions)
  return target;
}

size_t FlowPackagerCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.FlowPackagerCalculatorOptions)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    // optional string cache_file_format = 3 [default = "chunk_%04d"];
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_cache_file_format());
    }

    // optional .mediapipe.FlowPackagerOptions flow_packager_options = 1;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *flow_packager_options_);
    }

    // optional int32 caching_chunk_size_msec = 2 [default = 2500];
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_caching_chunk_size_msec());
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

void FlowPackagerCalculatorOptions::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.FlowPackagerCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  const FlowPackagerCalculatorOptions* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<FlowPackagerCalculatorOptions>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.FlowPackagerCalculatorOptions)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.FlowPackagerCalculatorOptions)
    MergeFrom(*source);
  }
}

void FlowPackagerCalculatorOptions::MergeFrom(const FlowPackagerCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.FlowPackagerCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    if (cached_has_bits & 0x00000001u) {
      _internal_set_cache_file_format(from._internal_cache_file_format());
    }
    if (cached_has_bits & 0x00000002u) {
      _internal_mutable_flow_packager_options()->::mediapipe::FlowPackagerOptions::MergeFrom(from._internal_flow_packager_options());
    }
    if (cached_has_bits & 0x00000004u) {
      caching_chunk_size_msec_ = from.caching_chunk_size_msec_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void FlowPackagerCalculatorOptions::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.FlowPackagerCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void FlowPackagerCalculatorOptions::CopyFrom(const FlowPackagerCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.FlowPackagerCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool FlowPackagerCalculatorOptions::IsInitialized() const {
  return true;
}

void FlowPackagerCalculatorOptions::InternalSwap(FlowPackagerCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  cache_file_format_.Swap(&other->cache_file_format_, nullptr, GetArena());
  swap(flow_packager_options_, other->flow_packager_options_);
  swap(caching_chunk_size_msec_, other->caching_chunk_size_msec_);
}

::PROTOBUF_NAMESPACE_ID::Metadata FlowPackagerCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2fvideo_2fflow_5fpackager_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int FlowPackagerCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::FlowPackagerCalculatorOptions >, 11, false >
  FlowPackagerCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::FlowPackagerCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::FlowPackagerCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::FlowPackagerCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::FlowPackagerCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <x/google/protobuf/port_undef.inc>
