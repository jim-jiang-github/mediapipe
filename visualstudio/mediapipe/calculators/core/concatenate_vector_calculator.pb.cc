// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/core/concatenate_vector_calculator.proto

#include "mediapipe/calculators/core/concatenate_vector_calculator.pb.h"

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
constexpr ConcatenateVectorCalculatorOptions::ConcatenateVectorCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : only_emit_if_all_present_(false){}
struct ConcatenateVectorCalculatorOptionsDefaultTypeInternal {
  constexpr ConcatenateVectorCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~ConcatenateVectorCalculatorOptionsDefaultTypeInternal() {}
  union {
    ConcatenateVectorCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT ConcatenateVectorCalculatorOptionsDefaultTypeInternal _ConcatenateVectorCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::ConcatenateVectorCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::ConcatenateVectorCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::ConcatenateVectorCalculatorOptions, only_emit_if_all_present_),
  0,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 6, sizeof(::mediapipe::ConcatenateVectorCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_ConcatenateVectorCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n>mediapipe/calculators/core/concatenate"
  "_vector_calculator.proto\022\tmediapipe\032$med"
  "iapipe/framework/calculator.proto\"\252\001\n\"Co"
  "ncatenateVectorCalculatorOptions\022\'\n\030only"
  "_emit_if_all_present\030\001 \001(\010:\005false2[\n\003ext"
  "\022\034.mediapipe.CalculatorOptions\030\317\261\330{ \001(\0132"
  "-.mediapipe.ConcatenateVectorCalculatorO"
  "ptionsB\014\242\002\tMediaPipe"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto_deps[1] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto = {
  false, false, 300, descriptor_table_protodef_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto, "mediapipe/calculators/core/concatenate_vector_calculator.proto", 
  &descriptor_table_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto_deps, 1, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto(&descriptor_table_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto);
namespace mediapipe {

// ===================================================================

class ConcatenateVectorCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<ConcatenateVectorCalculatorOptions>()._has_bits_);
  static void set_has_only_emit_if_all_present(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
};

ConcatenateVectorCalculatorOptions::ConcatenateVectorCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.ConcatenateVectorCalculatorOptions)
}
ConcatenateVectorCalculatorOptions::ConcatenateVectorCalculatorOptions(const ConcatenateVectorCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  only_emit_if_all_present_ = from.only_emit_if_all_present_;
  // @@protoc_insertion_point(copy_constructor:mediapipe.ConcatenateVectorCalculatorOptions)
}

void ConcatenateVectorCalculatorOptions::SharedCtor() {
only_emit_if_all_present_ = false;
}

ConcatenateVectorCalculatorOptions::~ConcatenateVectorCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.ConcatenateVectorCalculatorOptions)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void ConcatenateVectorCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void ConcatenateVectorCalculatorOptions::ArenaDtor(void* object) {
  ConcatenateVectorCalculatorOptions* _this = reinterpret_cast< ConcatenateVectorCalculatorOptions* >(object);
  (void)_this;
}
void ConcatenateVectorCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void ConcatenateVectorCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void ConcatenateVectorCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.ConcatenateVectorCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  only_emit_if_all_present_ = false;
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* ConcatenateVectorCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional bool only_emit_if_all_present = 1 [default = false];
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          _Internal::set_has_only_emit_if_all_present(&has_bits);
          only_emit_if_all_present_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
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

::PROTOBUF_NAMESPACE_ID::uint8* ConcatenateVectorCalculatorOptions::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.ConcatenateVectorCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional bool only_emit_if_all_present = 1 [default = false];
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(1, this->_internal_only_emit_if_all_present(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.ConcatenateVectorCalculatorOptions)
  return target;
}

size_t ConcatenateVectorCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.ConcatenateVectorCalculatorOptions)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // optional bool only_emit_if_all_present = 1 [default = false];
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000001u) {
    total_size += 1 + 1;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void ConcatenateVectorCalculatorOptions::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.ConcatenateVectorCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  const ConcatenateVectorCalculatorOptions* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<ConcatenateVectorCalculatorOptions>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.ConcatenateVectorCalculatorOptions)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.ConcatenateVectorCalculatorOptions)
    MergeFrom(*source);
  }
}

void ConcatenateVectorCalculatorOptions::MergeFrom(const ConcatenateVectorCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.ConcatenateVectorCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from._internal_has_only_emit_if_all_present()) {
    _internal_set_only_emit_if_all_present(from._internal_only_emit_if_all_present());
  }
}

void ConcatenateVectorCalculatorOptions::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.ConcatenateVectorCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void ConcatenateVectorCalculatorOptions::CopyFrom(const ConcatenateVectorCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.ConcatenateVectorCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ConcatenateVectorCalculatorOptions::IsInitialized() const {
  return true;
}

void ConcatenateVectorCalculatorOptions::InternalSwap(ConcatenateVectorCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  swap(only_emit_if_all_present_, other->only_emit_if_all_present_);
}

::PROTOBUF_NAMESPACE_ID::Metadata ConcatenateVectorCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2fcore_2fconcatenate_5fvector_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int ConcatenateVectorCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::ConcatenateVectorCalculatorOptions >, 11, false >
  ConcatenateVectorCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::ConcatenateVectorCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::ConcatenateVectorCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::ConcatenateVectorCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::ConcatenateVectorCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
