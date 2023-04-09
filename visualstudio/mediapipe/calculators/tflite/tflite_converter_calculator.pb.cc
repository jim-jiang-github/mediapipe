// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/tflite/tflite_converter_calculator.proto

#include "mediapipe/calculators/tflite/tflite_converter_calculator.pb.h"

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
constexpr TfLiteConverterCalculatorOptions_TensorFloatRange::TfLiteConverterCalculatorOptions_TensorFloatRange(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : min_(0)
  , max_(0){}
struct TfLiteConverterCalculatorOptions_TensorFloatRangeDefaultTypeInternal {
  constexpr TfLiteConverterCalculatorOptions_TensorFloatRangeDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~TfLiteConverterCalculatorOptions_TensorFloatRangeDefaultTypeInternal() {}
  union {
    TfLiteConverterCalculatorOptions_TensorFloatRange _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT TfLiteConverterCalculatorOptions_TensorFloatRangeDefaultTypeInternal _TfLiteConverterCalculatorOptions_TensorFloatRange_default_instance_;
constexpr TfLiteConverterCalculatorOptions::TfLiteConverterCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : output_tensor_float_range_(nullptr)
  , use_custom_normalization_(false)
  , flip_vertically_(false)
  , row_major_matrix_(false)
  , use_quantized_tensors_(false)
  , zero_center_(true)
  , max_num_channels_(3)
  , custom_div_(-1)
  , custom_sub_(-1){}
struct TfLiteConverterCalculatorOptionsDefaultTypeInternal {
  constexpr TfLiteConverterCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~TfLiteConverterCalculatorOptionsDefaultTypeInternal() {}
  union {
    TfLiteConverterCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT TfLiteConverterCalculatorOptionsDefaultTypeInternal _TfLiteConverterCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto[2];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange, min_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange, max_),
  0,
  1,
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteConverterCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteConverterCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteConverterCalculatorOptions, zero_center_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteConverterCalculatorOptions, use_custom_normalization_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteConverterCalculatorOptions, custom_div_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteConverterCalculatorOptions, custom_sub_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteConverterCalculatorOptions, flip_vertically_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteConverterCalculatorOptions, max_num_channels_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteConverterCalculatorOptions, row_major_matrix_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteConverterCalculatorOptions, use_quantized_tensors_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteConverterCalculatorOptions, output_tensor_float_range_),
  5,
  1,
  7,
  8,
  2,
  6,
  3,
  4,
  0,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 7, sizeof(::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange)},
  { 9, 23, sizeof(::mediapipe::TfLiteConverterCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_TfLiteConverterCalculatorOptions_TensorFloatRange_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_TfLiteConverterCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n>mediapipe/calculators/tflite/tflite_co"
  "nverter_calculator.proto\022\tmediapipe\032$med"
  "iapipe/framework/calculator.proto\"\204\004\n Tf"
  "LiteConverterCalculatorOptions\022\031\n\013zero_c"
  "enter\030\001 \001(\010:\004true\022\'\n\030use_custom_normaliz"
  "ation\030\006 \001(\010:\005false\022\026\n\ncustom_div\030\007 \001(\002:\002"
  "-1\022\026\n\ncustom_sub\030\010 \001(\002:\002-1\022\036\n\017flip_verti"
  "cally\030\002 \001(\010:\005false\022\033\n\020max_num_channels\030\003"
  " \001(\005:\0013\022\037\n\020row_major_matrix\030\004 \001(\010:\005false"
  "\022$\n\025use_quantized_tensors\030\005 \001(\010:\005false\022_"
  "\n\031output_tensor_float_range\030\t \001(\0132<.medi"
  "apipe.TfLiteConverterCalculatorOptions.T"
  "ensorFloatRange\032,\n\020TensorFloatRange\022\013\n\003m"
  "in\030\001 \001(\002\022\013\n\003max\030\002 \001(\0022Y\n\003ext\022\034.mediapipe"
  ".CalculatorOptions\030\305\303\233u \001(\0132+.mediapipe."
  "TfLiteConverterCalculatorOptions"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto_deps[1] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto = {
  false, false, 632, descriptor_table_protodef_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto, "mediapipe/calculators/tflite/tflite_converter_calculator.proto", 
  &descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto_deps, 1, 2,
  schemas, file_default_instances, TableStruct_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto(&descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto);
namespace mediapipe {

// ===================================================================

class TfLiteConverterCalculatorOptions_TensorFloatRange::_Internal {
 public:
  using HasBits = decltype(std::declval<TfLiteConverterCalculatorOptions_TensorFloatRange>()._has_bits_);
  static void set_has_min(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_max(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
};

TfLiteConverterCalculatorOptions_TensorFloatRange::TfLiteConverterCalculatorOptions_TensorFloatRange(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange)
}
TfLiteConverterCalculatorOptions_TensorFloatRange::TfLiteConverterCalculatorOptions_TensorFloatRange(const TfLiteConverterCalculatorOptions_TensorFloatRange& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&min_, &from.min_,
    static_cast<size_t>(reinterpret_cast<char*>(&max_) -
    reinterpret_cast<char*>(&min_)) + sizeof(max_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange)
}

void TfLiteConverterCalculatorOptions_TensorFloatRange::SharedCtor() {
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&min_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&max_) -
    reinterpret_cast<char*>(&min_)) + sizeof(max_));
}

TfLiteConverterCalculatorOptions_TensorFloatRange::~TfLiteConverterCalculatorOptions_TensorFloatRange() {
  // @@protoc_insertion_point(destructor:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void TfLiteConverterCalculatorOptions_TensorFloatRange::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void TfLiteConverterCalculatorOptions_TensorFloatRange::ArenaDtor(void* object) {
  TfLiteConverterCalculatorOptions_TensorFloatRange* _this = reinterpret_cast< TfLiteConverterCalculatorOptions_TensorFloatRange* >(object);
  (void)_this;
}
void TfLiteConverterCalculatorOptions_TensorFloatRange::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void TfLiteConverterCalculatorOptions_TensorFloatRange::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void TfLiteConverterCalculatorOptions_TensorFloatRange::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    ::memset(&min_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&max_) -
        reinterpret_cast<char*>(&min_)) + sizeof(max_));
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* TfLiteConverterCalculatorOptions_TensorFloatRange::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional float min = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 13)) {
          _Internal::set_has_min(&has_bits);
          min_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else goto handle_unusual;
        continue;
      // optional float max = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 21)) {
          _Internal::set_has_max(&has_bits);
          max_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
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

::PROTOBUF_NAMESPACE_ID::uint8* TfLiteConverterCalculatorOptions_TensorFloatRange::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional float min = 1;
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(1, this->_internal_min(), target);
  }

  // optional float max = 2;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(2, this->_internal_max(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange)
  return target;
}

size_t TfLiteConverterCalculatorOptions_TensorFloatRange::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    // optional float min = 1;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 + 4;
    }

    // optional float max = 2;
    if (cached_has_bits & 0x00000002u) {
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

void TfLiteConverterCalculatorOptions_TensorFloatRange::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange)
  GOOGLE_DCHECK_NE(&from, this);
  const TfLiteConverterCalculatorOptions_TensorFloatRange* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<TfLiteConverterCalculatorOptions_TensorFloatRange>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange)
    MergeFrom(*source);
  }
}

void TfLiteConverterCalculatorOptions_TensorFloatRange::MergeFrom(const TfLiteConverterCalculatorOptions_TensorFloatRange& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    if (cached_has_bits & 0x00000001u) {
      min_ = from.min_;
    }
    if (cached_has_bits & 0x00000002u) {
      max_ = from.max_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void TfLiteConverterCalculatorOptions_TensorFloatRange::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void TfLiteConverterCalculatorOptions_TensorFloatRange::CopyFrom(const TfLiteConverterCalculatorOptions_TensorFloatRange& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool TfLiteConverterCalculatorOptions_TensorFloatRange::IsInitialized() const {
  return true;
}

void TfLiteConverterCalculatorOptions_TensorFloatRange::InternalSwap(TfLiteConverterCalculatorOptions_TensorFloatRange* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(TfLiteConverterCalculatorOptions_TensorFloatRange, max_)
      + sizeof(TfLiteConverterCalculatorOptions_TensorFloatRange::max_)
      - PROTOBUF_FIELD_OFFSET(TfLiteConverterCalculatorOptions_TensorFloatRange, min_)>(
          reinterpret_cast<char*>(&min_),
          reinterpret_cast<char*>(&other->min_));
}

::PROTOBUF_NAMESPACE_ID::Metadata TfLiteConverterCalculatorOptions_TensorFloatRange::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto[0]);
}

// ===================================================================

class TfLiteConverterCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<TfLiteConverterCalculatorOptions>()._has_bits_);
  static void set_has_zero_center(HasBits* has_bits) {
    (*has_bits)[0] |= 32u;
  }
  static void set_has_use_custom_normalization(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static void set_has_custom_div(HasBits* has_bits) {
    (*has_bits)[0] |= 128u;
  }
  static void set_has_custom_sub(HasBits* has_bits) {
    (*has_bits)[0] |= 256u;
  }
  static void set_has_flip_vertically(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static void set_has_max_num_channels(HasBits* has_bits) {
    (*has_bits)[0] |= 64u;
  }
  static void set_has_row_major_matrix(HasBits* has_bits) {
    (*has_bits)[0] |= 8u;
  }
  static void set_has_use_quantized_tensors(HasBits* has_bits) {
    (*has_bits)[0] |= 16u;
  }
  static const ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange& output_tensor_float_range(const TfLiteConverterCalculatorOptions* msg);
  static void set_has_output_tensor_float_range(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
};

const ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange&
TfLiteConverterCalculatorOptions::_Internal::output_tensor_float_range(const TfLiteConverterCalculatorOptions* msg) {
  return *msg->output_tensor_float_range_;
}
TfLiteConverterCalculatorOptions::TfLiteConverterCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.TfLiteConverterCalculatorOptions)
}
TfLiteConverterCalculatorOptions::TfLiteConverterCalculatorOptions(const TfLiteConverterCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  if (from._internal_has_output_tensor_float_range()) {
    output_tensor_float_range_ = new ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange(*from.output_tensor_float_range_);
  } else {
    output_tensor_float_range_ = nullptr;
  }
  ::memcpy(&use_custom_normalization_, &from.use_custom_normalization_,
    static_cast<size_t>(reinterpret_cast<char*>(&custom_sub_) -
    reinterpret_cast<char*>(&use_custom_normalization_)) + sizeof(custom_sub_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.TfLiteConverterCalculatorOptions)
}

void TfLiteConverterCalculatorOptions::SharedCtor() {
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&output_tensor_float_range_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&use_quantized_tensors_) -
    reinterpret_cast<char*>(&output_tensor_float_range_)) + sizeof(use_quantized_tensors_));
zero_center_ = true;
max_num_channels_ = 3;
custom_div_ = -1;
custom_sub_ = -1;
}

TfLiteConverterCalculatorOptions::~TfLiteConverterCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.TfLiteConverterCalculatorOptions)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void TfLiteConverterCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
  if (this != internal_default_instance()) delete output_tensor_float_range_;
}

void TfLiteConverterCalculatorOptions::ArenaDtor(void* object) {
  TfLiteConverterCalculatorOptions* _this = reinterpret_cast< TfLiteConverterCalculatorOptions* >(object);
  (void)_this;
}
void TfLiteConverterCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void TfLiteConverterCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void TfLiteConverterCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.TfLiteConverterCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000001u) {
    GOOGLE_DCHECK(output_tensor_float_range_ != nullptr);
    output_tensor_float_range_->Clear();
  }
  ::memset(&use_custom_normalization_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&use_quantized_tensors_) -
      reinterpret_cast<char*>(&use_custom_normalization_)) + sizeof(use_quantized_tensors_));
  if (cached_has_bits & 0x000000e0u) {
    zero_center_ = true;
    max_num_channels_ = 3;
    custom_div_ = -1;
  }
  custom_sub_ = -1;
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* TfLiteConverterCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional bool zero_center = 1 [default = true];
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          _Internal::set_has_zero_center(&has_bits);
          zero_center_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional bool flip_vertically = 2 [default = false];
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          _Internal::set_has_flip_vertically(&has_bits);
          flip_vertically_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 max_num_channels = 3 [default = 3];
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          _Internal::set_has_max_num_channels(&has_bits);
          max_num_channels_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional bool row_major_matrix = 4 [default = false];
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 32)) {
          _Internal::set_has_row_major_matrix(&has_bits);
          row_major_matrix_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional bool use_quantized_tensors = 5 [default = false];
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 40)) {
          _Internal::set_has_use_quantized_tensors(&has_bits);
          use_quantized_tensors_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional bool use_custom_normalization = 6 [default = false];
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 48)) {
          _Internal::set_has_use_custom_normalization(&has_bits);
          use_custom_normalization_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional float custom_div = 7 [default = -1];
      case 7:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 61)) {
          _Internal::set_has_custom_div(&has_bits);
          custom_div_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else goto handle_unusual;
        continue;
      // optional float custom_sub = 8 [default = -1];
      case 8:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 69)) {
          _Internal::set_has_custom_sub(&has_bits);
          custom_sub_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else goto handle_unusual;
        continue;
      // optional .mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange output_tensor_float_range = 9;
      case 9:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 74)) {
          ptr = ctx->ParseMessage(_internal_mutable_output_tensor_float_range(), ptr);
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

::PROTOBUF_NAMESPACE_ID::uint8* TfLiteConverterCalculatorOptions::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.TfLiteConverterCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional bool zero_center = 1 [default = true];
  if (cached_has_bits & 0x00000020u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(1, this->_internal_zero_center(), target);
  }

  // optional bool flip_vertically = 2 [default = false];
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(2, this->_internal_flip_vertically(), target);
  }

  // optional int32 max_num_channels = 3 [default = 3];
  if (cached_has_bits & 0x00000040u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(3, this->_internal_max_num_channels(), target);
  }

  // optional bool row_major_matrix = 4 [default = false];
  if (cached_has_bits & 0x00000008u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(4, this->_internal_row_major_matrix(), target);
  }

  // optional bool use_quantized_tensors = 5 [default = false];
  if (cached_has_bits & 0x00000010u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(5, this->_internal_use_quantized_tensors(), target);
  }

  // optional bool use_custom_normalization = 6 [default = false];
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(6, this->_internal_use_custom_normalization(), target);
  }

  // optional float custom_div = 7 [default = -1];
  if (cached_has_bits & 0x00000080u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(7, this->_internal_custom_div(), target);
  }

  // optional float custom_sub = 8 [default = -1];
  if (cached_has_bits & 0x00000100u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(8, this->_internal_custom_sub(), target);
  }

  // optional .mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange output_tensor_float_range = 9;
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        9, _Internal::output_tensor_float_range(this), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.TfLiteConverterCalculatorOptions)
  return target;
}

size_t TfLiteConverterCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.TfLiteConverterCalculatorOptions)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    // optional .mediapipe.TfLiteConverterCalculatorOptions.TensorFloatRange output_tensor_float_range = 9;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *output_tensor_float_range_);
    }

    // optional bool use_custom_normalization = 6 [default = false];
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 + 1;
    }

    // optional bool flip_vertically = 2 [default = false];
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 + 1;
    }

    // optional bool row_major_matrix = 4 [default = false];
    if (cached_has_bits & 0x00000008u) {
      total_size += 1 + 1;
    }

    // optional bool use_quantized_tensors = 5 [default = false];
    if (cached_has_bits & 0x00000010u) {
      total_size += 1 + 1;
    }

    // optional bool zero_center = 1 [default = true];
    if (cached_has_bits & 0x00000020u) {
      total_size += 1 + 1;
    }

    // optional int32 max_num_channels = 3 [default = 3];
    if (cached_has_bits & 0x00000040u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_max_num_channels());
    }

    // optional float custom_div = 7 [default = -1];
    if (cached_has_bits & 0x00000080u) {
      total_size += 1 + 4;
    }

  }
  // optional float custom_sub = 8 [default = -1];
  if (cached_has_bits & 0x00000100u) {
    total_size += 1 + 4;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void TfLiteConverterCalculatorOptions::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.TfLiteConverterCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  const TfLiteConverterCalculatorOptions* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<TfLiteConverterCalculatorOptions>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.TfLiteConverterCalculatorOptions)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.TfLiteConverterCalculatorOptions)
    MergeFrom(*source);
  }
}

void TfLiteConverterCalculatorOptions::MergeFrom(const TfLiteConverterCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.TfLiteConverterCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    if (cached_has_bits & 0x00000001u) {
      _internal_mutable_output_tensor_float_range()->::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange::MergeFrom(from._internal_output_tensor_float_range());
    }
    if (cached_has_bits & 0x00000002u) {
      use_custom_normalization_ = from.use_custom_normalization_;
    }
    if (cached_has_bits & 0x00000004u) {
      flip_vertically_ = from.flip_vertically_;
    }
    if (cached_has_bits & 0x00000008u) {
      row_major_matrix_ = from.row_major_matrix_;
    }
    if (cached_has_bits & 0x00000010u) {
      use_quantized_tensors_ = from.use_quantized_tensors_;
    }
    if (cached_has_bits & 0x00000020u) {
      zero_center_ = from.zero_center_;
    }
    if (cached_has_bits & 0x00000040u) {
      max_num_channels_ = from.max_num_channels_;
    }
    if (cached_has_bits & 0x00000080u) {
      custom_div_ = from.custom_div_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
  if (cached_has_bits & 0x00000100u) {
    _internal_set_custom_sub(from._internal_custom_sub());
  }
}

void TfLiteConverterCalculatorOptions::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.TfLiteConverterCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void TfLiteConverterCalculatorOptions::CopyFrom(const TfLiteConverterCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.TfLiteConverterCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool TfLiteConverterCalculatorOptions::IsInitialized() const {
  return true;
}

void TfLiteConverterCalculatorOptions::InternalSwap(TfLiteConverterCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(TfLiteConverterCalculatorOptions, use_quantized_tensors_)
      + sizeof(TfLiteConverterCalculatorOptions::use_quantized_tensors_)
      - PROTOBUF_FIELD_OFFSET(TfLiteConverterCalculatorOptions, output_tensor_float_range_)>(
          reinterpret_cast<char*>(&output_tensor_float_range_),
          reinterpret_cast<char*>(&other->output_tensor_float_range_));
  swap(zero_center_, other->zero_center_);
  swap(max_num_channels_, other->max_num_channels_);
  swap(custom_div_, other->custom_div_);
  swap(custom_sub_, other->custom_sub_);
}

::PROTOBUF_NAMESPACE_ID::Metadata TfLiteConverterCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2ftflite_2ftflite_5fconverter_5fcalculator_2eproto[1]);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int TfLiteConverterCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::TfLiteConverterCalculatorOptions >, 11, false >
  TfLiteConverterCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::TfLiteConverterCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange* Arena::CreateMaybeMessage< ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::TfLiteConverterCalculatorOptions_TensorFloatRange >(arena);
}
template<> PROTOBUF_NOINLINE ::mediapipe::TfLiteConverterCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::TfLiteConverterCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::TfLiteConverterCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <x/google/protobuf/port_undef.inc>
