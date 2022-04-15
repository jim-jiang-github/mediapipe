// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/core/split_vector_calculator.proto

#include "mediapipe/calculators/core/split_vector_calculator.pb.h"

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
constexpr Range::Range(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : begin_(0)
  , end_(0){}
struct RangeDefaultTypeInternal {
  constexpr RangeDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~RangeDefaultTypeInternal() {}
  union {
    Range _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT RangeDefaultTypeInternal _Range_default_instance_;
constexpr SplitVectorCalculatorOptions::SplitVectorCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : ranges_()
  , element_only_(false)
  , combine_outputs_(false){}
struct SplitVectorCalculatorOptionsDefaultTypeInternal {
  constexpr SplitVectorCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~SplitVectorCalculatorOptionsDefaultTypeInternal() {}
  union {
    SplitVectorCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT SplitVectorCalculatorOptionsDefaultTypeInternal _SplitVectorCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto[2];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::Range, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::Range, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::Range, begin_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::Range, end_),
  0,
  1,
  PROTOBUF_FIELD_OFFSET(::mediapipe::SplitVectorCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SplitVectorCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::SplitVectorCalculatorOptions, ranges_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SplitVectorCalculatorOptions, element_only_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::SplitVectorCalculatorOptions, combine_outputs_),
  ~0u,
  0,
  1,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 7, sizeof(::mediapipe::Range)},
  { 9, 17, sizeof(::mediapipe::SplitVectorCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_Range_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_SplitVectorCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n8mediapipe/calculators/core/split_vecto"
  "r_calculator.proto\022\tmediapipe\032$mediapipe"
  "/framework/calculator.proto\"#\n\005Range\022\r\n\005"
  "begin\030\001 \001(\005\022\013\n\003end\030\002 \001(\005\"\324\001\n\034SplitVector"
  "CalculatorOptions\022 \n\006ranges\030\001 \003(\0132\020.medi"
  "apipe.Range\022\033\n\014element_only\030\002 \001(\010:\005false"
  "\022\036\n\017combine_outputs\030\003 \001(\010:\005false2U\n\003ext\022"
  "\034.mediapipe.CalculatorOptions\030\216\355\332{ \001(\0132\'"
  ".mediapipe.SplitVectorCalculatorOptionsB"
  "\014\242\002\tMediaPipe"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto_deps[1] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto = {
  false, false, 373, descriptor_table_protodef_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto, "mediapipe/calculators/core/split_vector_calculator.proto", 
  &descriptor_table_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto_deps, 1, 2,
  schemas, file_default_instances, TableStruct_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto(&descriptor_table_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto);
namespace mediapipe {

// ===================================================================

class Range::_Internal {
 public:
  using HasBits = decltype(std::declval<Range>()._has_bits_);
  static void set_has_begin(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_end(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
};

Range::Range(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.Range)
}
Range::Range(const Range& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&begin_, &from.begin_,
    static_cast<size_t>(reinterpret_cast<char*>(&end_) -
    reinterpret_cast<char*>(&begin_)) + sizeof(end_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.Range)
}

void Range::SharedCtor() {
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&begin_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&end_) -
    reinterpret_cast<char*>(&begin_)) + sizeof(end_));
}

Range::~Range() {
  // @@protoc_insertion_point(destructor:mediapipe.Range)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void Range::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void Range::ArenaDtor(void* object) {
  Range* _this = reinterpret_cast< Range* >(object);
  (void)_this;
}
void Range::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void Range::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void Range::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.Range)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    ::memset(&begin_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&end_) -
        reinterpret_cast<char*>(&begin_)) + sizeof(end_));
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* Range::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional int32 begin = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          _Internal::set_has_begin(&has_bits);
          begin_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 end = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          _Internal::set_has_end(&has_bits);
          end_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
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

::PROTOBUF_NAMESPACE_ID::uint8* Range::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.Range)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional int32 begin = 1;
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->_internal_begin(), target);
  }

  // optional int32 end = 2;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(2, this->_internal_end(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.Range)
  return target;
}

size_t Range::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.Range)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    // optional int32 begin = 1;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_begin());
    }

    // optional int32 end = 2;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_end());
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

void Range::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.Range)
  GOOGLE_DCHECK_NE(&from, this);
  const Range* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<Range>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.Range)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.Range)
    MergeFrom(*source);
  }
}

void Range::MergeFrom(const Range& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.Range)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    if (cached_has_bits & 0x00000001u) {
      begin_ = from.begin_;
    }
    if (cached_has_bits & 0x00000002u) {
      end_ = from.end_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void Range::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.Range)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Range::CopyFrom(const Range& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.Range)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Range::IsInitialized() const {
  return true;
}

void Range::InternalSwap(Range* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(Range, end_)
      + sizeof(Range::end_)
      - PROTOBUF_FIELD_OFFSET(Range, begin_)>(
          reinterpret_cast<char*>(&begin_),
          reinterpret_cast<char*>(&other->begin_));
}

::PROTOBUF_NAMESPACE_ID::Metadata Range::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto[0]);
}

// ===================================================================

class SplitVectorCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<SplitVectorCalculatorOptions>()._has_bits_);
  static void set_has_element_only(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_combine_outputs(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
};

SplitVectorCalculatorOptions::SplitVectorCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena),
  ranges_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.SplitVectorCalculatorOptions)
}
SplitVectorCalculatorOptions::SplitVectorCalculatorOptions(const SplitVectorCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_),
      ranges_(from.ranges_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&element_only_, &from.element_only_,
    static_cast<size_t>(reinterpret_cast<char*>(&combine_outputs_) -
    reinterpret_cast<char*>(&element_only_)) + sizeof(combine_outputs_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.SplitVectorCalculatorOptions)
}

void SplitVectorCalculatorOptions::SharedCtor() {
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&element_only_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&combine_outputs_) -
    reinterpret_cast<char*>(&element_only_)) + sizeof(combine_outputs_));
}

SplitVectorCalculatorOptions::~SplitVectorCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.SplitVectorCalculatorOptions)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void SplitVectorCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void SplitVectorCalculatorOptions::ArenaDtor(void* object) {
  SplitVectorCalculatorOptions* _this = reinterpret_cast< SplitVectorCalculatorOptions* >(object);
  (void)_this;
}
void SplitVectorCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void SplitVectorCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void SplitVectorCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.SplitVectorCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  ranges_.Clear();
  ::memset(&element_only_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&combine_outputs_) -
      reinterpret_cast<char*>(&element_only_)) + sizeof(combine_outputs_));
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* SplitVectorCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // repeated .mediapipe.Range ranges = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_ranges(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<10>(ptr));
        } else goto handle_unusual;
        continue;
      // optional bool element_only = 2 [default = false];
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          _Internal::set_has_element_only(&has_bits);
          element_only_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional bool combine_outputs = 3 [default = false];
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          _Internal::set_has_combine_outputs(&has_bits);
          combine_outputs_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
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

::PROTOBUF_NAMESPACE_ID::uint8* SplitVectorCalculatorOptions::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.SplitVectorCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .mediapipe.Range ranges = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->_internal_ranges_size()); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(1, this->_internal_ranges(i), target, stream);
  }

  cached_has_bits = _has_bits_[0];
  // optional bool element_only = 2 [default = false];
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(2, this->_internal_element_only(), target);
  }

  // optional bool combine_outputs = 3 [default = false];
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(3, this->_internal_combine_outputs(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.SplitVectorCalculatorOptions)
  return target;
}

size_t SplitVectorCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.SplitVectorCalculatorOptions)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .mediapipe.Range ranges = 1;
  total_size += 1UL * this->_internal_ranges_size();
  for (const auto& msg : this->ranges_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    // optional bool element_only = 2 [default = false];
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 + 1;
    }

    // optional bool combine_outputs = 3 [default = false];
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 + 1;
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

void SplitVectorCalculatorOptions::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.SplitVectorCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  const SplitVectorCalculatorOptions* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<SplitVectorCalculatorOptions>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.SplitVectorCalculatorOptions)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.SplitVectorCalculatorOptions)
    MergeFrom(*source);
  }
}

void SplitVectorCalculatorOptions::MergeFrom(const SplitVectorCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.SplitVectorCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  ranges_.MergeFrom(from.ranges_);
  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    if (cached_has_bits & 0x00000001u) {
      element_only_ = from.element_only_;
    }
    if (cached_has_bits & 0x00000002u) {
      combine_outputs_ = from.combine_outputs_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void SplitVectorCalculatorOptions::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.SplitVectorCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void SplitVectorCalculatorOptions::CopyFrom(const SplitVectorCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.SplitVectorCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool SplitVectorCalculatorOptions::IsInitialized() const {
  return true;
}

void SplitVectorCalculatorOptions::InternalSwap(SplitVectorCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  ranges_.InternalSwap(&other->ranges_);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(SplitVectorCalculatorOptions, combine_outputs_)
      + sizeof(SplitVectorCalculatorOptions::combine_outputs_)
      - PROTOBUF_FIELD_OFFSET(SplitVectorCalculatorOptions, element_only_)>(
          reinterpret_cast<char*>(&element_only_),
          reinterpret_cast<char*>(&other->element_only_));
}

::PROTOBUF_NAMESPACE_ID::Metadata SplitVectorCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2fcore_2fsplit_5fvector_5fcalculator_2eproto[1]);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int SplitVectorCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::SplitVectorCalculatorOptions >, 11, false >
  SplitVectorCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::SplitVectorCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::Range* Arena::CreateMaybeMessage< ::mediapipe::Range >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::Range >(arena);
}
template<> PROTOBUF_NOINLINE ::mediapipe::SplitVectorCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::SplitVectorCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::SplitVectorCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
