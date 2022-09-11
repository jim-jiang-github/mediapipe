// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/core/flow_limiter_calculator.proto

#include "mediapipe/calculators/core/flow_limiter_calculator.pb.h"

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
constexpr FlowLimiterCalculatorOptions::FlowLimiterCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : max_in_queue_(0)
  , max_in_flight_(1)
  , in_flight_timeout_(PROTOBUF_LONGLONG(1000000)){}
struct FlowLimiterCalculatorOptionsDefaultTypeInternal {
  constexpr FlowLimiterCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~FlowLimiterCalculatorOptionsDefaultTypeInternal() {}
  union {
    FlowLimiterCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT FlowLimiterCalculatorOptionsDefaultTypeInternal _FlowLimiterCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::FlowLimiterCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FlowLimiterCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::FlowLimiterCalculatorOptions, max_in_flight_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FlowLimiterCalculatorOptions, max_in_queue_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::FlowLimiterCalculatorOptions, in_flight_timeout_),
  1,
  0,
  2,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 8, sizeof(::mediapipe::FlowLimiterCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_FlowLimiterCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n8mediapipe/calculators/core/flow_limite"
  "r_calculator.proto\022\tmediapipe\032$mediapipe"
  "/framework/calculator.proto\"\315\001\n\034FlowLimi"
  "terCalculatorOptions\022\030\n\rmax_in_flight\030\001 "
  "\001(\005:\0011\022\027\n\014max_in_queue\030\002 \001(\005:\0010\022\"\n\021in_fl"
  "ight_timeout\030\003 \001(\003:\00710000002V\n\003ext\022\034.med"
  "iapipe.CalculatorOptions\030\370\240\364\233\001 \001(\0132\'.med"
  "iapipe.FlowLimiterCalculatorOptionsBC\n%c"
  "om.google.mediapipe.calculator.protoB\032Fl"
  "owLimiterCalculatorProto"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto_deps[1] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto = {
  false, false, 384, descriptor_table_protodef_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto, "mediapipe/calculators/core/flow_limiter_calculator.proto", 
  &descriptor_table_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto_deps, 1, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto(&descriptor_table_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto);
namespace mediapipe {

// ===================================================================

class FlowLimiterCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<FlowLimiterCalculatorOptions>()._has_bits_);
  static void set_has_max_in_flight(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static void set_has_max_in_queue(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_in_flight_timeout(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
};

FlowLimiterCalculatorOptions::FlowLimiterCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.FlowLimiterCalculatorOptions)
}
FlowLimiterCalculatorOptions::FlowLimiterCalculatorOptions(const FlowLimiterCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&max_in_queue_, &from.max_in_queue_,
    static_cast<size_t>(reinterpret_cast<char*>(&in_flight_timeout_) -
    reinterpret_cast<char*>(&max_in_queue_)) + sizeof(in_flight_timeout_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.FlowLimiterCalculatorOptions)
}

void FlowLimiterCalculatorOptions::SharedCtor() {
max_in_queue_ = 0;
max_in_flight_ = 1;
in_flight_timeout_ = PROTOBUF_LONGLONG(1000000);
}

FlowLimiterCalculatorOptions::~FlowLimiterCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.FlowLimiterCalculatorOptions)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void FlowLimiterCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void FlowLimiterCalculatorOptions::ArenaDtor(void* object) {
  FlowLimiterCalculatorOptions* _this = reinterpret_cast< FlowLimiterCalculatorOptions* >(object);
  (void)_this;
}
void FlowLimiterCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void FlowLimiterCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void FlowLimiterCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.FlowLimiterCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    max_in_queue_ = 0;
    max_in_flight_ = 1;
    in_flight_timeout_ = PROTOBUF_LONGLONG(1000000);
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* FlowLimiterCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional int32 max_in_flight = 1 [default = 1];
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          _Internal::set_has_max_in_flight(&has_bits);
          max_in_flight_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 max_in_queue = 2 [default = 0];
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          _Internal::set_has_max_in_queue(&has_bits);
          max_in_queue_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int64 in_flight_timeout = 3 [default = 1000000];
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          _Internal::set_has_in_flight_timeout(&has_bits);
          in_flight_timeout_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
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

::PROTOBUF_NAMESPACE_ID::uint8* FlowLimiterCalculatorOptions::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.FlowLimiterCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional int32 max_in_flight = 1 [default = 1];
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->_internal_max_in_flight(), target);
  }

  // optional int32 max_in_queue = 2 [default = 0];
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(2, this->_internal_max_in_queue(), target);
  }

  // optional int64 in_flight_timeout = 3 [default = 1000000];
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt64ToArray(3, this->_internal_in_flight_timeout(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.FlowLimiterCalculatorOptions)
  return target;
}

size_t FlowLimiterCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.FlowLimiterCalculatorOptions)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    // optional int32 max_in_queue = 2 [default = 0];
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_max_in_queue());
    }

    // optional int32 max_in_flight = 1 [default = 1];
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_max_in_flight());
    }

    // optional int64 in_flight_timeout = 3 [default = 1000000];
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int64Size(
          this->_internal_in_flight_timeout());
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

void FlowLimiterCalculatorOptions::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.FlowLimiterCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  const FlowLimiterCalculatorOptions* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<FlowLimiterCalculatorOptions>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.FlowLimiterCalculatorOptions)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.FlowLimiterCalculatorOptions)
    MergeFrom(*source);
  }
}

void FlowLimiterCalculatorOptions::MergeFrom(const FlowLimiterCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.FlowLimiterCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    if (cached_has_bits & 0x00000001u) {
      max_in_queue_ = from.max_in_queue_;
    }
    if (cached_has_bits & 0x00000002u) {
      max_in_flight_ = from.max_in_flight_;
    }
    if (cached_has_bits & 0x00000004u) {
      in_flight_timeout_ = from.in_flight_timeout_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void FlowLimiterCalculatorOptions::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.FlowLimiterCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void FlowLimiterCalculatorOptions::CopyFrom(const FlowLimiterCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.FlowLimiterCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool FlowLimiterCalculatorOptions::IsInitialized() const {
  return true;
}

void FlowLimiterCalculatorOptions::InternalSwap(FlowLimiterCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  swap(max_in_queue_, other->max_in_queue_);
  swap(max_in_flight_, other->max_in_flight_);
  swap(in_flight_timeout_, other->in_flight_timeout_);
}

::PROTOBUF_NAMESPACE_ID::Metadata FlowLimiterCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2fcore_2fflow_5flimiter_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int FlowLimiterCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::FlowLimiterCalculatorOptions >, 11, false >
  FlowLimiterCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::FlowLimiterCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::FlowLimiterCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::FlowLimiterCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::FlowLimiterCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
