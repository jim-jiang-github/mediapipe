// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/graphs/iris_tracking/calculators/iris_to_depth_calculator.proto

#include "mediapipe/graphs/iris_tracking/calculators/iris_to_depth_calculator.pb.h"

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
constexpr IrisToDepthCalculatorOptions::IrisToDepthCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : left_iris_center_index_(0)
  , right_iris_right_index_(8)
  , left_iris_top_index_(2)
  , left_iris_bottom_index_(4)
  , left_iris_left_index_(3)
  , left_iris_right_index_(1)
  , right_iris_center_index_(5)
  , right_iris_top_index_(7)
  , right_iris_bottom_index_(9)
  , right_iris_left_index_(6){}
struct IrisToDepthCalculatorOptionsDefaultTypeInternal {
  constexpr IrisToDepthCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~IrisToDepthCalculatorOptionsDefaultTypeInternal() {}
  union {
    IrisToDepthCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT IrisToDepthCalculatorOptionsDefaultTypeInternal _IrisToDepthCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToDepthCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToDepthCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToDepthCalculatorOptions, left_iris_center_index_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToDepthCalculatorOptions, left_iris_top_index_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToDepthCalculatorOptions, left_iris_bottom_index_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToDepthCalculatorOptions, left_iris_left_index_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToDepthCalculatorOptions, left_iris_right_index_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToDepthCalculatorOptions, right_iris_center_index_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToDepthCalculatorOptions, right_iris_top_index_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToDepthCalculatorOptions, right_iris_bottom_index_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToDepthCalculatorOptions, right_iris_left_index_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToDepthCalculatorOptions, right_iris_right_index_),
  0,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  1,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 15, sizeof(::mediapipe::IrisToDepthCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_IrisToDepthCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\nImediapipe/graphs/iris_tracking/calcula"
  "tors/iris_to_depth_calculator.proto\022\tmed"
  "iapipe\032$mediapipe/framework/calculator.p"
  "roto\"\315\003\n\034IrisToDepthCalculatorOptions\022!\n"
  "\026left_iris_center_index\030\001 \001(\005:\0010\022\036\n\023left"
  "_iris_top_index\030\002 \001(\005:\0012\022!\n\026left_iris_bo"
  "ttom_index\030\003 \001(\005:\0014\022\037\n\024left_iris_left_in"
  "dex\030\004 \001(\005:\0013\022 \n\025left_iris_right_index\030\005 "
  "\001(\005:\0011\022\"\n\027right_iris_center_index\030\006 \001(\005:"
  "\0015\022\037\n\024right_iris_top_index\030\007 \001(\005:\0017\022\"\n\027r"
  "ight_iris_bottom_index\030\010 \001(\005:\0019\022 \n\025right"
  "_iris_left_index\030\t \001(\005:\0016\022!\n\026right_iris_"
  "right_index\030\n \001(\005:\00182V\n\003ext\022\034.mediapipe."
  "CalculatorOptions\030\212\353\327\220\001 \001(\0132\'.mediapipe."
  "IrisToDepthCalculatorOptions"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto_deps[1] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto = {
  false, false, 588, descriptor_table_protodef_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto, "mediapipe/graphs/iris_tracking/calculators/iris_to_depth_calculator.proto", 
  &descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto_deps, 1, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto(&descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto);
namespace mediapipe {

// ===================================================================

class IrisToDepthCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<IrisToDepthCalculatorOptions>()._has_bits_);
  static void set_has_left_iris_center_index(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_left_iris_top_index(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static void set_has_left_iris_bottom_index(HasBits* has_bits) {
    (*has_bits)[0] |= 8u;
  }
  static void set_has_left_iris_left_index(HasBits* has_bits) {
    (*has_bits)[0] |= 16u;
  }
  static void set_has_left_iris_right_index(HasBits* has_bits) {
    (*has_bits)[0] |= 32u;
  }
  static void set_has_right_iris_center_index(HasBits* has_bits) {
    (*has_bits)[0] |= 64u;
  }
  static void set_has_right_iris_top_index(HasBits* has_bits) {
    (*has_bits)[0] |= 128u;
  }
  static void set_has_right_iris_bottom_index(HasBits* has_bits) {
    (*has_bits)[0] |= 256u;
  }
  static void set_has_right_iris_left_index(HasBits* has_bits) {
    (*has_bits)[0] |= 512u;
  }
  static void set_has_right_iris_right_index(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
};

IrisToDepthCalculatorOptions::IrisToDepthCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.IrisToDepthCalculatorOptions)
}
IrisToDepthCalculatorOptions::IrisToDepthCalculatorOptions(const IrisToDepthCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&left_iris_center_index_, &from.left_iris_center_index_,
    static_cast<size_t>(reinterpret_cast<char*>(&right_iris_left_index_) -
    reinterpret_cast<char*>(&left_iris_center_index_)) + sizeof(right_iris_left_index_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.IrisToDepthCalculatorOptions)
}

void IrisToDepthCalculatorOptions::SharedCtor() {
left_iris_center_index_ = 0;
right_iris_right_index_ = 8;
left_iris_top_index_ = 2;
left_iris_bottom_index_ = 4;
left_iris_left_index_ = 3;
left_iris_right_index_ = 1;
right_iris_center_index_ = 5;
right_iris_top_index_ = 7;
right_iris_bottom_index_ = 9;
right_iris_left_index_ = 6;
}

IrisToDepthCalculatorOptions::~IrisToDepthCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.IrisToDepthCalculatorOptions)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void IrisToDepthCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void IrisToDepthCalculatorOptions::ArenaDtor(void* object) {
  IrisToDepthCalculatorOptions* _this = reinterpret_cast< IrisToDepthCalculatorOptions* >(object);
  (void)_this;
}
void IrisToDepthCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void IrisToDepthCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void IrisToDepthCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.IrisToDepthCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    left_iris_center_index_ = 0;
    right_iris_right_index_ = 8;
    left_iris_top_index_ = 2;
    left_iris_bottom_index_ = 4;
    left_iris_left_index_ = 3;
    left_iris_right_index_ = 1;
    right_iris_center_index_ = 5;
    right_iris_top_index_ = 7;
  }
  if (cached_has_bits & 0x00000300u) {
    right_iris_bottom_index_ = 9;
    right_iris_left_index_ = 6;
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* IrisToDepthCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional int32 left_iris_center_index = 1 [default = 0];
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          _Internal::set_has_left_iris_center_index(&has_bits);
          left_iris_center_index_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 left_iris_top_index = 2 [default = 2];
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          _Internal::set_has_left_iris_top_index(&has_bits);
          left_iris_top_index_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 left_iris_bottom_index = 3 [default = 4];
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          _Internal::set_has_left_iris_bottom_index(&has_bits);
          left_iris_bottom_index_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 left_iris_left_index = 4 [default = 3];
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 32)) {
          _Internal::set_has_left_iris_left_index(&has_bits);
          left_iris_left_index_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 left_iris_right_index = 5 [default = 1];
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 40)) {
          _Internal::set_has_left_iris_right_index(&has_bits);
          left_iris_right_index_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 right_iris_center_index = 6 [default = 5];
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 48)) {
          _Internal::set_has_right_iris_center_index(&has_bits);
          right_iris_center_index_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 right_iris_top_index = 7 [default = 7];
      case 7:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 56)) {
          _Internal::set_has_right_iris_top_index(&has_bits);
          right_iris_top_index_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 right_iris_bottom_index = 8 [default = 9];
      case 8:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 64)) {
          _Internal::set_has_right_iris_bottom_index(&has_bits);
          right_iris_bottom_index_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 right_iris_left_index = 9 [default = 6];
      case 9:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 72)) {
          _Internal::set_has_right_iris_left_index(&has_bits);
          right_iris_left_index_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 right_iris_right_index = 10 [default = 8];
      case 10:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 80)) {
          _Internal::set_has_right_iris_right_index(&has_bits);
          right_iris_right_index_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
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

::PROTOBUF_NAMESPACE_ID::uint8* IrisToDepthCalculatorOptions::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.IrisToDepthCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional int32 left_iris_center_index = 1 [default = 0];
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->_internal_left_iris_center_index(), target);
  }

  // optional int32 left_iris_top_index = 2 [default = 2];
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(2, this->_internal_left_iris_top_index(), target);
  }

  // optional int32 left_iris_bottom_index = 3 [default = 4];
  if (cached_has_bits & 0x00000008u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(3, this->_internal_left_iris_bottom_index(), target);
  }

  // optional int32 left_iris_left_index = 4 [default = 3];
  if (cached_has_bits & 0x00000010u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(4, this->_internal_left_iris_left_index(), target);
  }

  // optional int32 left_iris_right_index = 5 [default = 1];
  if (cached_has_bits & 0x00000020u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(5, this->_internal_left_iris_right_index(), target);
  }

  // optional int32 right_iris_center_index = 6 [default = 5];
  if (cached_has_bits & 0x00000040u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(6, this->_internal_right_iris_center_index(), target);
  }

  // optional int32 right_iris_top_index = 7 [default = 7];
  if (cached_has_bits & 0x00000080u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(7, this->_internal_right_iris_top_index(), target);
  }

  // optional int32 right_iris_bottom_index = 8 [default = 9];
  if (cached_has_bits & 0x00000100u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(8, this->_internal_right_iris_bottom_index(), target);
  }

  // optional int32 right_iris_left_index = 9 [default = 6];
  if (cached_has_bits & 0x00000200u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(9, this->_internal_right_iris_left_index(), target);
  }

  // optional int32 right_iris_right_index = 10 [default = 8];
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(10, this->_internal_right_iris_right_index(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.IrisToDepthCalculatorOptions)
  return target;
}

size_t IrisToDepthCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.IrisToDepthCalculatorOptions)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    // optional int32 left_iris_center_index = 1 [default = 0];
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_left_iris_center_index());
    }

    // optional int32 right_iris_right_index = 10 [default = 8];
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_right_iris_right_index());
    }

    // optional int32 left_iris_top_index = 2 [default = 2];
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_left_iris_top_index());
    }

    // optional int32 left_iris_bottom_index = 3 [default = 4];
    if (cached_has_bits & 0x00000008u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_left_iris_bottom_index());
    }

    // optional int32 left_iris_left_index = 4 [default = 3];
    if (cached_has_bits & 0x00000010u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_left_iris_left_index());
    }

    // optional int32 left_iris_right_index = 5 [default = 1];
    if (cached_has_bits & 0x00000020u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_left_iris_right_index());
    }

    // optional int32 right_iris_center_index = 6 [default = 5];
    if (cached_has_bits & 0x00000040u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_right_iris_center_index());
    }

    // optional int32 right_iris_top_index = 7 [default = 7];
    if (cached_has_bits & 0x00000080u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_right_iris_top_index());
    }

  }
  if (cached_has_bits & 0x00000300u) {
    // optional int32 right_iris_bottom_index = 8 [default = 9];
    if (cached_has_bits & 0x00000100u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_right_iris_bottom_index());
    }

    // optional int32 right_iris_left_index = 9 [default = 6];
    if (cached_has_bits & 0x00000200u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_right_iris_left_index());
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

void IrisToDepthCalculatorOptions::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.IrisToDepthCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  const IrisToDepthCalculatorOptions* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<IrisToDepthCalculatorOptions>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.IrisToDepthCalculatorOptions)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.IrisToDepthCalculatorOptions)
    MergeFrom(*source);
  }
}

void IrisToDepthCalculatorOptions::MergeFrom(const IrisToDepthCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.IrisToDepthCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    if (cached_has_bits & 0x00000001u) {
      left_iris_center_index_ = from.left_iris_center_index_;
    }
    if (cached_has_bits & 0x00000002u) {
      right_iris_right_index_ = from.right_iris_right_index_;
    }
    if (cached_has_bits & 0x00000004u) {
      left_iris_top_index_ = from.left_iris_top_index_;
    }
    if (cached_has_bits & 0x00000008u) {
      left_iris_bottom_index_ = from.left_iris_bottom_index_;
    }
    if (cached_has_bits & 0x00000010u) {
      left_iris_left_index_ = from.left_iris_left_index_;
    }
    if (cached_has_bits & 0x00000020u) {
      left_iris_right_index_ = from.left_iris_right_index_;
    }
    if (cached_has_bits & 0x00000040u) {
      right_iris_center_index_ = from.right_iris_center_index_;
    }
    if (cached_has_bits & 0x00000080u) {
      right_iris_top_index_ = from.right_iris_top_index_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
  if (cached_has_bits & 0x00000300u) {
    if (cached_has_bits & 0x00000100u) {
      right_iris_bottom_index_ = from.right_iris_bottom_index_;
    }
    if (cached_has_bits & 0x00000200u) {
      right_iris_left_index_ = from.right_iris_left_index_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void IrisToDepthCalculatorOptions::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.IrisToDepthCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void IrisToDepthCalculatorOptions::CopyFrom(const IrisToDepthCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.IrisToDepthCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool IrisToDepthCalculatorOptions::IsInitialized() const {
  return true;
}

void IrisToDepthCalculatorOptions::InternalSwap(IrisToDepthCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  swap(left_iris_center_index_, other->left_iris_center_index_);
  swap(right_iris_right_index_, other->right_iris_right_index_);
  swap(left_iris_top_index_, other->left_iris_top_index_);
  swap(left_iris_bottom_index_, other->left_iris_bottom_index_);
  swap(left_iris_left_index_, other->left_iris_left_index_);
  swap(left_iris_right_index_, other->left_iris_right_index_);
  swap(right_iris_center_index_, other->right_iris_center_index_);
  swap(right_iris_top_index_, other->right_iris_top_index_);
  swap(right_iris_bottom_index_, other->right_iris_bottom_index_);
  swap(right_iris_left_index_, other->right_iris_left_index_);
}

::PROTOBUF_NAMESPACE_ID::Metadata IrisToDepthCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5fdepth_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int IrisToDepthCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::IrisToDepthCalculatorOptions >, 11, false >
  IrisToDepthCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::IrisToDepthCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::IrisToDepthCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::IrisToDepthCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::IrisToDepthCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <x/google/protobuf/port_undef.inc>
