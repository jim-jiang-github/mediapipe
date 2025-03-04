// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/util/refine_landmarks_from_heatmap_calculator.proto

#include "mediapipe/calculators/util/refine_landmarks_from_heatmap_calculator.pb.h"

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
constexpr RefineLandmarksFromHeatmapCalculatorOptions::RefineLandmarksFromHeatmapCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : refine_presence_(false)
  , refine_visibility_(false)
  , kernel_size_(9)
  , min_confidence_to_refine_(0.5f){}
struct RefineLandmarksFromHeatmapCalculatorOptionsDefaultTypeInternal {
  constexpr RefineLandmarksFromHeatmapCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~RefineLandmarksFromHeatmapCalculatorOptionsDefaultTypeInternal() {}
  union {
    RefineLandmarksFromHeatmapCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT RefineLandmarksFromHeatmapCalculatorOptionsDefaultTypeInternal _RefineLandmarksFromHeatmapCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::RefineLandmarksFromHeatmapCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::RefineLandmarksFromHeatmapCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::RefineLandmarksFromHeatmapCalculatorOptions, kernel_size_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::RefineLandmarksFromHeatmapCalculatorOptions, min_confidence_to_refine_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::RefineLandmarksFromHeatmapCalculatorOptions, refine_presence_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::RefineLandmarksFromHeatmapCalculatorOptions, refine_visibility_),
  2,
  3,
  0,
  1,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 9, sizeof(::mediapipe::RefineLandmarksFromHeatmapCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_RefineLandmarksFromHeatmapCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\nImediapipe/calculators/util/refine_land"
  "marks_from_heatmap_calculator.proto\022\tmed"
  "iapipe\032$mediapipe/framework/calculator.p"
  "roto\"\225\002\n+RefineLandmarksFromHeatmapCalcu"
  "latorOptions\022\026\n\013kernel_size\030\001 \001(\005:\0019\022%\n\030"
  "min_confidence_to_refine\030\002 \001(\002:\0030.5\022\036\n\017r"
  "efine_presence\030\003 \001(\010:\005false\022 \n\021refine_vi"
  "sibility\030\004 \001(\010:\005false2e\n\003ext\022\034.mediapipe"
  ".CalculatorOptions\030\265\365\337\254\001 \001(\01326.mediapipe"
  ".RefineLandmarksFromHeatmapCalculatorOpt"
  "ions"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto_deps[1] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto = {
  false, false, 404, descriptor_table_protodef_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto, "mediapipe/calculators/util/refine_landmarks_from_heatmap_calculator.proto", 
  &descriptor_table_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto_deps, 1, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto(&descriptor_table_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto);
namespace mediapipe {

// ===================================================================

class RefineLandmarksFromHeatmapCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<RefineLandmarksFromHeatmapCalculatorOptions>()._has_bits_);
  static void set_has_kernel_size(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static void set_has_min_confidence_to_refine(HasBits* has_bits) {
    (*has_bits)[0] |= 8u;
  }
  static void set_has_refine_presence(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_refine_visibility(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
};

RefineLandmarksFromHeatmapCalculatorOptions::RefineLandmarksFromHeatmapCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.RefineLandmarksFromHeatmapCalculatorOptions)
}
RefineLandmarksFromHeatmapCalculatorOptions::RefineLandmarksFromHeatmapCalculatorOptions(const RefineLandmarksFromHeatmapCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&refine_presence_, &from.refine_presence_,
    static_cast<size_t>(reinterpret_cast<char*>(&min_confidence_to_refine_) -
    reinterpret_cast<char*>(&refine_presence_)) + sizeof(min_confidence_to_refine_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.RefineLandmarksFromHeatmapCalculatorOptions)
}

void RefineLandmarksFromHeatmapCalculatorOptions::SharedCtor() {
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&refine_presence_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&refine_visibility_) -
    reinterpret_cast<char*>(&refine_presence_)) + sizeof(refine_visibility_));
kernel_size_ = 9;
min_confidence_to_refine_ = 0.5f;
}

RefineLandmarksFromHeatmapCalculatorOptions::~RefineLandmarksFromHeatmapCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.RefineLandmarksFromHeatmapCalculatorOptions)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void RefineLandmarksFromHeatmapCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void RefineLandmarksFromHeatmapCalculatorOptions::ArenaDtor(void* object) {
  RefineLandmarksFromHeatmapCalculatorOptions* _this = reinterpret_cast< RefineLandmarksFromHeatmapCalculatorOptions* >(object);
  (void)_this;
}
void RefineLandmarksFromHeatmapCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void RefineLandmarksFromHeatmapCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void RefineLandmarksFromHeatmapCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.RefineLandmarksFromHeatmapCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  ::memset(&refine_presence_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&refine_visibility_) -
      reinterpret_cast<char*>(&refine_presence_)) + sizeof(refine_visibility_));
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x0000000cu) {
    kernel_size_ = 9;
    min_confidence_to_refine_ = 0.5f;
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* RefineLandmarksFromHeatmapCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional int32 kernel_size = 1 [default = 9];
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          _Internal::set_has_kernel_size(&has_bits);
          kernel_size_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional float min_confidence_to_refine = 2 [default = 0.5];
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 21)) {
          _Internal::set_has_min_confidence_to_refine(&has_bits);
          min_confidence_to_refine_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else goto handle_unusual;
        continue;
      // optional bool refine_presence = 3 [default = false];
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          _Internal::set_has_refine_presence(&has_bits);
          refine_presence_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional bool refine_visibility = 4 [default = false];
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 32)) {
          _Internal::set_has_refine_visibility(&has_bits);
          refine_visibility_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
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

::PROTOBUF_NAMESPACE_ID::uint8* RefineLandmarksFromHeatmapCalculatorOptions::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.RefineLandmarksFromHeatmapCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional int32 kernel_size = 1 [default = 9];
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->_internal_kernel_size(), target);
  }

  // optional float min_confidence_to_refine = 2 [default = 0.5];
  if (cached_has_bits & 0x00000008u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(2, this->_internal_min_confidence_to_refine(), target);
  }

  // optional bool refine_presence = 3 [default = false];
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(3, this->_internal_refine_presence(), target);
  }

  // optional bool refine_visibility = 4 [default = false];
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(4, this->_internal_refine_visibility(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.RefineLandmarksFromHeatmapCalculatorOptions)
  return target;
}

size_t RefineLandmarksFromHeatmapCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.RefineLandmarksFromHeatmapCalculatorOptions)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x0000000fu) {
    // optional bool refine_presence = 3 [default = false];
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 + 1;
    }

    // optional bool refine_visibility = 4 [default = false];
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 + 1;
    }

    // optional int32 kernel_size = 1 [default = 9];
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_kernel_size());
    }

    // optional float min_confidence_to_refine = 2 [default = 0.5];
    if (cached_has_bits & 0x00000008u) {
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

void RefineLandmarksFromHeatmapCalculatorOptions::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.RefineLandmarksFromHeatmapCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  const RefineLandmarksFromHeatmapCalculatorOptions* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<RefineLandmarksFromHeatmapCalculatorOptions>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.RefineLandmarksFromHeatmapCalculatorOptions)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.RefineLandmarksFromHeatmapCalculatorOptions)
    MergeFrom(*source);
  }
}

void RefineLandmarksFromHeatmapCalculatorOptions::MergeFrom(const RefineLandmarksFromHeatmapCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.RefineLandmarksFromHeatmapCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x0000000fu) {
    if (cached_has_bits & 0x00000001u) {
      refine_presence_ = from.refine_presence_;
    }
    if (cached_has_bits & 0x00000002u) {
      refine_visibility_ = from.refine_visibility_;
    }
    if (cached_has_bits & 0x00000004u) {
      kernel_size_ = from.kernel_size_;
    }
    if (cached_has_bits & 0x00000008u) {
      min_confidence_to_refine_ = from.min_confidence_to_refine_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void RefineLandmarksFromHeatmapCalculatorOptions::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.RefineLandmarksFromHeatmapCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void RefineLandmarksFromHeatmapCalculatorOptions::CopyFrom(const RefineLandmarksFromHeatmapCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.RefineLandmarksFromHeatmapCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool RefineLandmarksFromHeatmapCalculatorOptions::IsInitialized() const {
  return true;
}

void RefineLandmarksFromHeatmapCalculatorOptions::InternalSwap(RefineLandmarksFromHeatmapCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(RefineLandmarksFromHeatmapCalculatorOptions, refine_visibility_)
      + sizeof(RefineLandmarksFromHeatmapCalculatorOptions::refine_visibility_)
      - PROTOBUF_FIELD_OFFSET(RefineLandmarksFromHeatmapCalculatorOptions, refine_presence_)>(
          reinterpret_cast<char*>(&refine_presence_),
          reinterpret_cast<char*>(&other->refine_presence_));
  swap(kernel_size_, other->kernel_size_);
  swap(min_confidence_to_refine_, other->min_confidence_to_refine_);
}

::PROTOBUF_NAMESPACE_ID::Metadata RefineLandmarksFromHeatmapCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2futil_2frefine_5flandmarks_5ffrom_5fheatmap_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int RefineLandmarksFromHeatmapCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::RefineLandmarksFromHeatmapCalculatorOptions >, 11, false >
  RefineLandmarksFromHeatmapCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::RefineLandmarksFromHeatmapCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::RefineLandmarksFromHeatmapCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::RefineLandmarksFromHeatmapCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::RefineLandmarksFromHeatmapCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
