// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/image/warp_affine_calculator.proto

#include "mediapipe/calculators/image/warp_affine_calculator.pb.h"

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
constexpr WarpAffineCalculatorOptions::WarpAffineCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : border_mode_(0)

  , gpu_origin_(0)
{}
struct WarpAffineCalculatorOptionsDefaultTypeInternal {
  constexpr WarpAffineCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~WarpAffineCalculatorOptionsDefaultTypeInternal() {}
  union {
    WarpAffineCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT WarpAffineCalculatorOptionsDefaultTypeInternal _WarpAffineCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto[1];
static const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* file_level_enum_descriptors_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::WarpAffineCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::WarpAffineCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::WarpAffineCalculatorOptions, border_mode_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::WarpAffineCalculatorOptions, gpu_origin_),
  0,
  1,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 7, sizeof(::mediapipe::WarpAffineCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_WarpAffineCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n8mediapipe/calculators/image/warp_affin"
  "e_calculator.proto\022\tmediapipe\032$mediapipe"
  "/framework/calculator.proto\032\036mediapipe/g"
  "pu/gpu_origin.proto\"\270\002\n\033WarpAffineCalcul"
  "atorOptions\022F\n\013border_mode\030\001 \001(\01621.media"
  "pipe.WarpAffineCalculatorOptions.BorderM"
  "ode\022-\n\ngpu_origin\030\002 \001(\0162\031.mediapipe.GpuO"
  "rigin.Mode\"K\n\nBorderMode\022\026\n\022BORDER_UNSPE"
  "CIFIED\020\000\022\017\n\013BORDER_ZERO\020\001\022\024\n\020BORDER_REPL"
  "ICATE\020\0022U\n\003ext\022\034.mediapipe.CalculatorOpt"
  "ions\030\307\273\230\262\001 \001(\0132&.mediapipe.WarpAffineCal"
  "culatorOptions"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto_deps[2] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
  &::descriptor_table_mediapipe_2fgpu_2fgpu_5forigin_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto = {
  false, false, 454, descriptor_table_protodef_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto, "mediapipe/calculators/image/warp_affine_calculator.proto", 
  &descriptor_table_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto_deps, 2, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto(&descriptor_table_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto);
namespace mediapipe {
const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* WarpAffineCalculatorOptions_BorderMode_descriptor() {
  ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&descriptor_table_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto);
  return file_level_enum_descriptors_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto[0];
}
bool WarpAffineCalculatorOptions_BorderMode_IsValid(int value) {
  switch (value) {
    case 0:
    case 1:
    case 2:
      return true;
    default:
      return false;
  }
}

#if (__cplusplus < 201703) && (!defined(_MSC_VER) || _MSC_VER >= 1900)
constexpr WarpAffineCalculatorOptions_BorderMode WarpAffineCalculatorOptions::BORDER_UNSPECIFIED;
constexpr WarpAffineCalculatorOptions_BorderMode WarpAffineCalculatorOptions::BORDER_ZERO;
constexpr WarpAffineCalculatorOptions_BorderMode WarpAffineCalculatorOptions::BORDER_REPLICATE;
constexpr WarpAffineCalculatorOptions_BorderMode WarpAffineCalculatorOptions::BorderMode_MIN;
constexpr WarpAffineCalculatorOptions_BorderMode WarpAffineCalculatorOptions::BorderMode_MAX;
constexpr int WarpAffineCalculatorOptions::BorderMode_ARRAYSIZE;
#endif  // (__cplusplus < 201703) && (!defined(_MSC_VER) || _MSC_VER >= 1900)

// ===================================================================

class WarpAffineCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<WarpAffineCalculatorOptions>()._has_bits_);
  static void set_has_border_mode(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_gpu_origin(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
};

WarpAffineCalculatorOptions::WarpAffineCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.WarpAffineCalculatorOptions)
}
WarpAffineCalculatorOptions::WarpAffineCalculatorOptions(const WarpAffineCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&border_mode_, &from.border_mode_,
    static_cast<size_t>(reinterpret_cast<char*>(&gpu_origin_) -
    reinterpret_cast<char*>(&border_mode_)) + sizeof(gpu_origin_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.WarpAffineCalculatorOptions)
}

void WarpAffineCalculatorOptions::SharedCtor() {
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&border_mode_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&gpu_origin_) -
    reinterpret_cast<char*>(&border_mode_)) + sizeof(gpu_origin_));
}

WarpAffineCalculatorOptions::~WarpAffineCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.WarpAffineCalculatorOptions)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void WarpAffineCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void WarpAffineCalculatorOptions::ArenaDtor(void* object) {
  WarpAffineCalculatorOptions* _this = reinterpret_cast< WarpAffineCalculatorOptions* >(object);
  (void)_this;
}
void WarpAffineCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void WarpAffineCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void WarpAffineCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.WarpAffineCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    ::memset(&border_mode_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&gpu_origin_) -
        reinterpret_cast<char*>(&border_mode_)) + sizeof(gpu_origin_));
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* WarpAffineCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional .mediapipe.WarpAffineCalculatorOptions.BorderMode border_mode = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          ::PROTOBUF_NAMESPACE_ID::uint64 val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
          if (PROTOBUF_PREDICT_TRUE(::mediapipe::WarpAffineCalculatorOptions_BorderMode_IsValid(val))) {
            _internal_set_border_mode(static_cast<::mediapipe::WarpAffineCalculatorOptions_BorderMode>(val));
          } else {
            ::PROTOBUF_NAMESPACE_ID::internal::WriteVarint(1, val, mutable_unknown_fields());
          }
        } else goto handle_unusual;
        continue;
      // optional .mediapipe.GpuOrigin.Mode gpu_origin = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          ::PROTOBUF_NAMESPACE_ID::uint64 val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
          if (PROTOBUF_PREDICT_TRUE(::mediapipe::GpuOrigin_Mode_IsValid(val))) {
            _internal_set_gpu_origin(static_cast<::mediapipe::GpuOrigin_Mode>(val));
          } else {
            ::PROTOBUF_NAMESPACE_ID::internal::WriteVarint(2, val, mutable_unknown_fields());
          }
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

::PROTOBUF_NAMESPACE_ID::uint8* WarpAffineCalculatorOptions::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.WarpAffineCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional .mediapipe.WarpAffineCalculatorOptions.BorderMode border_mode = 1;
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteEnumToArray(
      1, this->_internal_border_mode(), target);
  }

  // optional .mediapipe.GpuOrigin.Mode gpu_origin = 2;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteEnumToArray(
      2, this->_internal_gpu_origin(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.WarpAffineCalculatorOptions)
  return target;
}

size_t WarpAffineCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.WarpAffineCalculatorOptions)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    // optional .mediapipe.WarpAffineCalculatorOptions.BorderMode border_mode = 1;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::EnumSize(this->_internal_border_mode());
    }

    // optional .mediapipe.GpuOrigin.Mode gpu_origin = 2;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::EnumSize(this->_internal_gpu_origin());
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

void WarpAffineCalculatorOptions::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.WarpAffineCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  const WarpAffineCalculatorOptions* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<WarpAffineCalculatorOptions>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.WarpAffineCalculatorOptions)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.WarpAffineCalculatorOptions)
    MergeFrom(*source);
  }
}

void WarpAffineCalculatorOptions::MergeFrom(const WarpAffineCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.WarpAffineCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    if (cached_has_bits & 0x00000001u) {
      border_mode_ = from.border_mode_;
    }
    if (cached_has_bits & 0x00000002u) {
      gpu_origin_ = from.gpu_origin_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void WarpAffineCalculatorOptions::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.WarpAffineCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void WarpAffineCalculatorOptions::CopyFrom(const WarpAffineCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.WarpAffineCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool WarpAffineCalculatorOptions::IsInitialized() const {
  return true;
}

void WarpAffineCalculatorOptions::InternalSwap(WarpAffineCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(WarpAffineCalculatorOptions, gpu_origin_)
      + sizeof(WarpAffineCalculatorOptions::gpu_origin_)
      - PROTOBUF_FIELD_OFFSET(WarpAffineCalculatorOptions, border_mode_)>(
          reinterpret_cast<char*>(&border_mode_),
          reinterpret_cast<char*>(&other->border_mode_));
}

::PROTOBUF_NAMESPACE_ID::Metadata WarpAffineCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2fimage_2fwarp_5faffine_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int WarpAffineCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::WarpAffineCalculatorOptions >, 11, false >
  WarpAffineCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::WarpAffineCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::WarpAffineCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::WarpAffineCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::WarpAffineCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
