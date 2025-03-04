// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/framework/formats/annotation/rasterization.proto

#include "mediapipe/framework/formats/annotation/rasterization.pb.h"

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
constexpr Rasterization_Interval::Rasterization_Interval(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : y_(0)
  , left_x_(0)
  , right_x_(0){}
struct Rasterization_IntervalDefaultTypeInternal {
  constexpr Rasterization_IntervalDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~Rasterization_IntervalDefaultTypeInternal() {}
  union {
    Rasterization_Interval _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT Rasterization_IntervalDefaultTypeInternal _Rasterization_Interval_default_instance_;
constexpr Rasterization::Rasterization(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : interval_(){}
struct RasterizationDefaultTypeInternal {
  constexpr RasterizationDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~RasterizationDefaultTypeInternal() {}
  union {
    Rasterization _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT RasterizationDefaultTypeInternal _Rasterization_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto[2];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::Rasterization_Interval, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::Rasterization_Interval, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::Rasterization_Interval, y_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::Rasterization_Interval, left_x_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::Rasterization_Interval, right_x_),
  0,
  1,
  2,
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::mediapipe::Rasterization, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::Rasterization, interval_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 8, sizeof(::mediapipe::Rasterization_Interval)},
  { 11, -1, sizeof(::mediapipe::Rasterization)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_Rasterization_Interval_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_Rasterization_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n:mediapipe/framework/formats/annotation"
  "/rasterization.proto\022\tmediapipe\"|\n\rRaste"
  "rization\0223\n\010interval\030\001 \003(\0132!.mediapipe.R"
  "asterization.Interval\0326\n\010Interval\022\t\n\001y\030\001"
  " \002(\005\022\016\n\006left_x\030\002 \002(\005\022\017\n\007right_x\030\003 \002(\005BC\n"
  "-com.google.mediapipe.formats.annotation"
  ".protoB\022RasterizationProto"
  ;
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto = {
  false, false, 266, descriptor_table_protodef_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto, "mediapipe/framework/formats/annotation/rasterization.proto", 
  &descriptor_table_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto_once, nullptr, 0, 2,
  schemas, file_default_instances, TableStruct_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto::offsets,
  file_level_metadata_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto, file_level_enum_descriptors_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto, file_level_service_descriptors_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto_getter() {
  return &descriptor_table_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto(&descriptor_table_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto);
namespace mediapipe {

// ===================================================================

class Rasterization_Interval::_Internal {
 public:
  using HasBits = decltype(std::declval<Rasterization_Interval>()._has_bits_);
  static void set_has_y(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_left_x(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static void set_has_right_x(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static bool MissingRequiredFields(const HasBits& has_bits) {
    return ((has_bits[0] & 0x00000007) ^ 0x00000007) != 0;
  }
};

Rasterization_Interval::Rasterization_Interval(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.Rasterization.Interval)
}
Rasterization_Interval::Rasterization_Interval(const Rasterization_Interval& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&y_, &from.y_,
    static_cast<size_t>(reinterpret_cast<char*>(&right_x_) -
    reinterpret_cast<char*>(&y_)) + sizeof(right_x_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.Rasterization.Interval)
}

void Rasterization_Interval::SharedCtor() {
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&y_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&right_x_) -
    reinterpret_cast<char*>(&y_)) + sizeof(right_x_));
}

Rasterization_Interval::~Rasterization_Interval() {
  // @@protoc_insertion_point(destructor:mediapipe.Rasterization.Interval)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void Rasterization_Interval::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void Rasterization_Interval::ArenaDtor(void* object) {
  Rasterization_Interval* _this = reinterpret_cast< Rasterization_Interval* >(object);
  (void)_this;
}
void Rasterization_Interval::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void Rasterization_Interval::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void Rasterization_Interval::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.Rasterization.Interval)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    ::memset(&y_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&right_x_) -
        reinterpret_cast<char*>(&y_)) + sizeof(right_x_));
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* Rasterization_Interval::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // required int32 y = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          _Internal::set_has_y(&has_bits);
          y_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // required int32 left_x = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          _Internal::set_has_left_x(&has_bits);
          left_x_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // required int32 right_x = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          _Internal::set_has_right_x(&has_bits);
          right_x_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
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

::PROTOBUF_NAMESPACE_ID::uint8* Rasterization_Interval::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.Rasterization.Interval)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // required int32 y = 1;
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->_internal_y(), target);
  }

  // required int32 left_x = 2;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(2, this->_internal_left_x(), target);
  }

  // required int32 right_x = 3;
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(3, this->_internal_right_x(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.Rasterization.Interval)
  return target;
}

size_t Rasterization_Interval::RequiredFieldsByteSizeFallback() const {
// @@protoc_insertion_point(required_fields_byte_size_fallback_start:mediapipe.Rasterization.Interval)
  size_t total_size = 0;

  if (_internal_has_y()) {
    // required int32 y = 1;
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
        this->_internal_y());
  }

  if (_internal_has_left_x()) {
    // required int32 left_x = 2;
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
        this->_internal_left_x());
  }

  if (_internal_has_right_x()) {
    // required int32 right_x = 3;
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
        this->_internal_right_x());
  }

  return total_size;
}
size_t Rasterization_Interval::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.Rasterization.Interval)
  size_t total_size = 0;

  if (((_has_bits_[0] & 0x00000007) ^ 0x00000007) == 0) {  // All required fields are present.
    // required int32 y = 1;
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
        this->_internal_y());

    // required int32 left_x = 2;
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
        this->_internal_left_x());

    // required int32 right_x = 3;
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
        this->_internal_right_x());

  } else {
    total_size += RequiredFieldsByteSizeFallback();
  }
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void Rasterization_Interval::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.Rasterization.Interval)
  GOOGLE_DCHECK_NE(&from, this);
  const Rasterization_Interval* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<Rasterization_Interval>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.Rasterization.Interval)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.Rasterization.Interval)
    MergeFrom(*source);
  }
}

void Rasterization_Interval::MergeFrom(const Rasterization_Interval& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.Rasterization.Interval)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    if (cached_has_bits & 0x00000001u) {
      y_ = from.y_;
    }
    if (cached_has_bits & 0x00000002u) {
      left_x_ = from.left_x_;
    }
    if (cached_has_bits & 0x00000004u) {
      right_x_ = from.right_x_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void Rasterization_Interval::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.Rasterization.Interval)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Rasterization_Interval::CopyFrom(const Rasterization_Interval& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.Rasterization.Interval)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Rasterization_Interval::IsInitialized() const {
  if (_Internal::MissingRequiredFields(_has_bits_)) return false;
  return true;
}

void Rasterization_Interval::InternalSwap(Rasterization_Interval* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(Rasterization_Interval, right_x_)
      + sizeof(Rasterization_Interval::right_x_)
      - PROTOBUF_FIELD_OFFSET(Rasterization_Interval, y_)>(
          reinterpret_cast<char*>(&y_),
          reinterpret_cast<char*>(&other->y_));
}

::PROTOBUF_NAMESPACE_ID::Metadata Rasterization_Interval::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto_getter, &descriptor_table_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto_once,
      file_level_metadata_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto[0]);
}

// ===================================================================

class Rasterization::_Internal {
 public:
};

Rasterization::Rasterization(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena),
  interval_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.Rasterization)
}
Rasterization::Rasterization(const Rasterization& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      interval_(from.interval_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:mediapipe.Rasterization)
}

void Rasterization::SharedCtor() {
}

Rasterization::~Rasterization() {
  // @@protoc_insertion_point(destructor:mediapipe.Rasterization)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void Rasterization::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void Rasterization::ArenaDtor(void* object) {
  Rasterization* _this = reinterpret_cast< Rasterization* >(object);
  (void)_this;
}
void Rasterization::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void Rasterization::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void Rasterization::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.Rasterization)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  interval_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* Rasterization::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // repeated .mediapipe.Rasterization.Interval interval = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_interval(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<10>(ptr));
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
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* Rasterization::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.Rasterization)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .mediapipe.Rasterization.Interval interval = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->_internal_interval_size()); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(1, this->_internal_interval(i), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.Rasterization)
  return target;
}

size_t Rasterization::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.Rasterization)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .mediapipe.Rasterization.Interval interval = 1;
  total_size += 1UL * this->_internal_interval_size();
  for (const auto& msg : this->interval_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void Rasterization::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.Rasterization)
  GOOGLE_DCHECK_NE(&from, this);
  const Rasterization* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<Rasterization>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.Rasterization)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.Rasterization)
    MergeFrom(*source);
  }
}

void Rasterization::MergeFrom(const Rasterization& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.Rasterization)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  interval_.MergeFrom(from.interval_);
}

void Rasterization::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.Rasterization)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Rasterization::CopyFrom(const Rasterization& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.Rasterization)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Rasterization::IsInitialized() const {
  if (!::PROTOBUF_NAMESPACE_ID::internal::AllAreInitialized(interval_)) return false;
  return true;
}

void Rasterization::InternalSwap(Rasterization* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  interval_.InternalSwap(&other->interval_);
}

::PROTOBUF_NAMESPACE_ID::Metadata Rasterization::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto_getter, &descriptor_table_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto_once,
      file_level_metadata_mediapipe_2fframework_2fformats_2fannotation_2frasterization_2eproto[1]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::Rasterization_Interval* Arena::CreateMaybeMessage< ::mediapipe::Rasterization_Interval >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::Rasterization_Interval >(arena);
}
template<> PROTOBUF_NOINLINE ::mediapipe::Rasterization* Arena::CreateMaybeMessage< ::mediapipe::Rasterization >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::Rasterization >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
