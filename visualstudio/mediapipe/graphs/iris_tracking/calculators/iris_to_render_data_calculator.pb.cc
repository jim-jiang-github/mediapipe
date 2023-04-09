// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/graphs/iris_tracking/calculators/iris_to_render_data_calculator.proto

#include "mediapipe/graphs/iris_tracking/calculators/iris_to_render_data_calculator.pb.h"

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
constexpr IrisToRenderDataCalculatorOptions::IrisToRenderDataCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : oval_color_(nullptr)
  , landmark_color_(nullptr)
  , font_face_(0)
  , location_(0)

  , horizontal_offset_px_(0)
  , vertical_offset_px_(0)
  , font_height_px_(50)
  , oval_thickness_(1)
  , landmark_thickness_(1){}
struct IrisToRenderDataCalculatorOptionsDefaultTypeInternal {
  constexpr IrisToRenderDataCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~IrisToRenderDataCalculatorOptionsDefaultTypeInternal() {}
  union {
    IrisToRenderDataCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT IrisToRenderDataCalculatorOptionsDefaultTypeInternal _IrisToRenderDataCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto[1];
static const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* file_level_enum_descriptors_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToRenderDataCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToRenderDataCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToRenderDataCalculatorOptions, oval_color_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToRenderDataCalculatorOptions, landmark_color_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToRenderDataCalculatorOptions, oval_thickness_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToRenderDataCalculatorOptions, landmark_thickness_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToRenderDataCalculatorOptions, font_height_px_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToRenderDataCalculatorOptions, horizontal_offset_px_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToRenderDataCalculatorOptions, vertical_offset_px_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToRenderDataCalculatorOptions, font_face_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::IrisToRenderDataCalculatorOptions, location_),
  0,
  1,
  7,
  8,
  6,
  4,
  5,
  2,
  3,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 14, sizeof(::mediapipe::IrisToRenderDataCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_IrisToRenderDataCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\nOmediapipe/graphs/iris_tracking/calcula"
  "tors/iris_to_render_data_calculator.prot"
  "o\022\tmediapipe\032$mediapipe/framework/calcul"
  "ator.proto\032\032mediapipe/util/color.proto\"\372"
  "\003\n!IrisToRenderDataCalculatorOptions\022$\n\n"
  "oval_color\030\001 \001(\0132\020.mediapipe.Color\022(\n\016la"
  "ndmark_color\030\t \001(\0132\020.mediapipe.Color\022\031\n\016"
  "oval_thickness\030\002 \001(\001:\0011\022\035\n\022landmark_thic"
  "kness\030\n \001(\001:\0011\022\032\n\016font_height_px\030\003 \001(\005:\002"
  "50\022\037\n\024horizontal_offset_px\030\007 \001(\005:\0010\022\035\n\022v"
  "ertical_offset_px\030\010 \001(\005:\0010\022\024\n\tfont_face\030"
  "\005 \001(\005:\0010\022Q\n\010location\030\006 \001(\01625.mediapipe.I"
  "risToRenderDataCalculatorOptions.Locatio"
  "n:\010TOP_LEFT\")\n\010Location\022\014\n\010TOP_LEFT\020\000\022\017\n"
  "\013BOTTOM_LEFT\020\0012[\n\003ext\022\034.mediapipe.Calcul"
  "atorOptions\030\270\301\207\212\001 \001(\0132,.mediapipe.IrisTo"
  "RenderDataCalculatorOptions"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto_deps[2] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
  &::descriptor_table_mediapipe_2futil_2fcolor_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto = {
  false, false, 667, descriptor_table_protodef_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto, "mediapipe/graphs/iris_tracking/calculators/iris_to_render_data_calculator.proto", 
  &descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto_deps, 2, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto(&descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto);
namespace mediapipe {
const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* IrisToRenderDataCalculatorOptions_Location_descriptor() {
  ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto);
  return file_level_enum_descriptors_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto[0];
}
bool IrisToRenderDataCalculatorOptions_Location_IsValid(int value) {
  switch (value) {
    case 0:
    case 1:
      return true;
    default:
      return false;
  }
}

#if (__cplusplus < 201703) && (!defined(_MSC_VER) || _MSC_VER >= 1900)
constexpr IrisToRenderDataCalculatorOptions_Location IrisToRenderDataCalculatorOptions::TOP_LEFT;
constexpr IrisToRenderDataCalculatorOptions_Location IrisToRenderDataCalculatorOptions::BOTTOM_LEFT;
constexpr IrisToRenderDataCalculatorOptions_Location IrisToRenderDataCalculatorOptions::Location_MIN;
constexpr IrisToRenderDataCalculatorOptions_Location IrisToRenderDataCalculatorOptions::Location_MAX;
constexpr int IrisToRenderDataCalculatorOptions::Location_ARRAYSIZE;
#endif  // (__cplusplus < 201703) && (!defined(_MSC_VER) || _MSC_VER >= 1900)

// ===================================================================

class IrisToRenderDataCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<IrisToRenderDataCalculatorOptions>()._has_bits_);
  static const ::mediapipe::Color& oval_color(const IrisToRenderDataCalculatorOptions* msg);
  static void set_has_oval_color(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static const ::mediapipe::Color& landmark_color(const IrisToRenderDataCalculatorOptions* msg);
  static void set_has_landmark_color(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static void set_has_oval_thickness(HasBits* has_bits) {
    (*has_bits)[0] |= 128u;
  }
  static void set_has_landmark_thickness(HasBits* has_bits) {
    (*has_bits)[0] |= 256u;
  }
  static void set_has_font_height_px(HasBits* has_bits) {
    (*has_bits)[0] |= 64u;
  }
  static void set_has_horizontal_offset_px(HasBits* has_bits) {
    (*has_bits)[0] |= 16u;
  }
  static void set_has_vertical_offset_px(HasBits* has_bits) {
    (*has_bits)[0] |= 32u;
  }
  static void set_has_font_face(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static void set_has_location(HasBits* has_bits) {
    (*has_bits)[0] |= 8u;
  }
};

const ::mediapipe::Color&
IrisToRenderDataCalculatorOptions::_Internal::oval_color(const IrisToRenderDataCalculatorOptions* msg) {
  return *msg->oval_color_;
}
const ::mediapipe::Color&
IrisToRenderDataCalculatorOptions::_Internal::landmark_color(const IrisToRenderDataCalculatorOptions* msg) {
  return *msg->landmark_color_;
}
void IrisToRenderDataCalculatorOptions::clear_oval_color() {
  if (oval_color_ != nullptr) oval_color_->Clear();
  _has_bits_[0] &= ~0x00000001u;
}
void IrisToRenderDataCalculatorOptions::clear_landmark_color() {
  if (landmark_color_ != nullptr) landmark_color_->Clear();
  _has_bits_[0] &= ~0x00000002u;
}
IrisToRenderDataCalculatorOptions::IrisToRenderDataCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.IrisToRenderDataCalculatorOptions)
}
IrisToRenderDataCalculatorOptions::IrisToRenderDataCalculatorOptions(const IrisToRenderDataCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  if (from._internal_has_oval_color()) {
    oval_color_ = new ::mediapipe::Color(*from.oval_color_);
  } else {
    oval_color_ = nullptr;
  }
  if (from._internal_has_landmark_color()) {
    landmark_color_ = new ::mediapipe::Color(*from.landmark_color_);
  } else {
    landmark_color_ = nullptr;
  }
  ::memcpy(&font_face_, &from.font_face_,
    static_cast<size_t>(reinterpret_cast<char*>(&landmark_thickness_) -
    reinterpret_cast<char*>(&font_face_)) + sizeof(landmark_thickness_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.IrisToRenderDataCalculatorOptions)
}

void IrisToRenderDataCalculatorOptions::SharedCtor() {
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&oval_color_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&vertical_offset_px_) -
    reinterpret_cast<char*>(&oval_color_)) + sizeof(vertical_offset_px_));
font_height_px_ = 50;
oval_thickness_ = 1;
landmark_thickness_ = 1;
}

IrisToRenderDataCalculatorOptions::~IrisToRenderDataCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.IrisToRenderDataCalculatorOptions)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void IrisToRenderDataCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
  if (this != internal_default_instance()) delete oval_color_;
  if (this != internal_default_instance()) delete landmark_color_;
}

void IrisToRenderDataCalculatorOptions::ArenaDtor(void* object) {
  IrisToRenderDataCalculatorOptions* _this = reinterpret_cast< IrisToRenderDataCalculatorOptions* >(object);
  (void)_this;
}
void IrisToRenderDataCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void IrisToRenderDataCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void IrisToRenderDataCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.IrisToRenderDataCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    if (cached_has_bits & 0x00000001u) {
      GOOGLE_DCHECK(oval_color_ != nullptr);
      oval_color_->Clear();
    }
    if (cached_has_bits & 0x00000002u) {
      GOOGLE_DCHECK(landmark_color_ != nullptr);
      landmark_color_->Clear();
    }
  }
  if (cached_has_bits & 0x000000fcu) {
    ::memset(&font_face_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&vertical_offset_px_) -
        reinterpret_cast<char*>(&font_face_)) + sizeof(vertical_offset_px_));
    font_height_px_ = 50;
    oval_thickness_ = 1;
  }
  landmark_thickness_ = 1;
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* IrisToRenderDataCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional .mediapipe.Color oval_color = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr = ctx->ParseMessage(_internal_mutable_oval_color(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional double oval_thickness = 2 [default = 1];
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 17)) {
          _Internal::set_has_oval_thickness(&has_bits);
          oval_thickness_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr);
          ptr += sizeof(double);
        } else goto handle_unusual;
        continue;
      // optional int32 font_height_px = 3 [default = 50];
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          _Internal::set_has_font_height_px(&has_bits);
          font_height_px_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 font_face = 5 [default = 0];
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 40)) {
          _Internal::set_has_font_face(&has_bits);
          font_face_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional .mediapipe.IrisToRenderDataCalculatorOptions.Location location = 6 [default = TOP_LEFT];
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 48)) {
          ::PROTOBUF_NAMESPACE_ID::uint64 val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
          if (PROTOBUF_PREDICT_TRUE(::mediapipe::IrisToRenderDataCalculatorOptions_Location_IsValid(val))) {
            _internal_set_location(static_cast<::mediapipe::IrisToRenderDataCalculatorOptions_Location>(val));
          } else {
            ::PROTOBUF_NAMESPACE_ID::internal::WriteVarint(6, val, mutable_unknown_fields());
          }
        } else goto handle_unusual;
        continue;
      // optional int32 horizontal_offset_px = 7 [default = 0];
      case 7:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 56)) {
          _Internal::set_has_horizontal_offset_px(&has_bits);
          horizontal_offset_px_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 vertical_offset_px = 8 [default = 0];
      case 8:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 64)) {
          _Internal::set_has_vertical_offset_px(&has_bits);
          vertical_offset_px_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional .mediapipe.Color landmark_color = 9;
      case 9:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 74)) {
          ptr = ctx->ParseMessage(_internal_mutable_landmark_color(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional double landmark_thickness = 10 [default = 1];
      case 10:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 81)) {
          _Internal::set_has_landmark_thickness(&has_bits);
          landmark_thickness_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr);
          ptr += sizeof(double);
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

::PROTOBUF_NAMESPACE_ID::uint8* IrisToRenderDataCalculatorOptions::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.IrisToRenderDataCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional .mediapipe.Color oval_color = 1;
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        1, _Internal::oval_color(this), target, stream);
  }

  // optional double oval_thickness = 2 [default = 1];
  if (cached_has_bits & 0x00000080u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteDoubleToArray(2, this->_internal_oval_thickness(), target);
  }

  // optional int32 font_height_px = 3 [default = 50];
  if (cached_has_bits & 0x00000040u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(3, this->_internal_font_height_px(), target);
  }

  // optional int32 font_face = 5 [default = 0];
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(5, this->_internal_font_face(), target);
  }

  // optional .mediapipe.IrisToRenderDataCalculatorOptions.Location location = 6 [default = TOP_LEFT];
  if (cached_has_bits & 0x00000008u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteEnumToArray(
      6, this->_internal_location(), target);
  }

  // optional int32 horizontal_offset_px = 7 [default = 0];
  if (cached_has_bits & 0x00000010u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(7, this->_internal_horizontal_offset_px(), target);
  }

  // optional int32 vertical_offset_px = 8 [default = 0];
  if (cached_has_bits & 0x00000020u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(8, this->_internal_vertical_offset_px(), target);
  }

  // optional .mediapipe.Color landmark_color = 9;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        9, _Internal::landmark_color(this), target, stream);
  }

  // optional double landmark_thickness = 10 [default = 1];
  if (cached_has_bits & 0x00000100u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteDoubleToArray(10, this->_internal_landmark_thickness(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.IrisToRenderDataCalculatorOptions)
  return target;
}

size_t IrisToRenderDataCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.IrisToRenderDataCalculatorOptions)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    // optional .mediapipe.Color oval_color = 1;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *oval_color_);
    }

    // optional .mediapipe.Color landmark_color = 9;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *landmark_color_);
    }

    // optional int32 font_face = 5 [default = 0];
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_font_face());
    }

    // optional .mediapipe.IrisToRenderDataCalculatorOptions.Location location = 6 [default = TOP_LEFT];
    if (cached_has_bits & 0x00000008u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::EnumSize(this->_internal_location());
    }

    // optional int32 horizontal_offset_px = 7 [default = 0];
    if (cached_has_bits & 0x00000010u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_horizontal_offset_px());
    }

    // optional int32 vertical_offset_px = 8 [default = 0];
    if (cached_has_bits & 0x00000020u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_vertical_offset_px());
    }

    // optional int32 font_height_px = 3 [default = 50];
    if (cached_has_bits & 0x00000040u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_font_height_px());
    }

    // optional double oval_thickness = 2 [default = 1];
    if (cached_has_bits & 0x00000080u) {
      total_size += 1 + 8;
    }

  }
  // optional double landmark_thickness = 10 [default = 1];
  if (cached_has_bits & 0x00000100u) {
    total_size += 1 + 8;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void IrisToRenderDataCalculatorOptions::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.IrisToRenderDataCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  const IrisToRenderDataCalculatorOptions* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<IrisToRenderDataCalculatorOptions>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.IrisToRenderDataCalculatorOptions)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.IrisToRenderDataCalculatorOptions)
    MergeFrom(*source);
  }
}

void IrisToRenderDataCalculatorOptions::MergeFrom(const IrisToRenderDataCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.IrisToRenderDataCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    if (cached_has_bits & 0x00000001u) {
      _internal_mutable_oval_color()->::mediapipe::Color::MergeFrom(from._internal_oval_color());
    }
    if (cached_has_bits & 0x00000002u) {
      _internal_mutable_landmark_color()->::mediapipe::Color::MergeFrom(from._internal_landmark_color());
    }
    if (cached_has_bits & 0x00000004u) {
      font_face_ = from.font_face_;
    }
    if (cached_has_bits & 0x00000008u) {
      location_ = from.location_;
    }
    if (cached_has_bits & 0x00000010u) {
      horizontal_offset_px_ = from.horizontal_offset_px_;
    }
    if (cached_has_bits & 0x00000020u) {
      vertical_offset_px_ = from.vertical_offset_px_;
    }
    if (cached_has_bits & 0x00000040u) {
      font_height_px_ = from.font_height_px_;
    }
    if (cached_has_bits & 0x00000080u) {
      oval_thickness_ = from.oval_thickness_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
  if (cached_has_bits & 0x00000100u) {
    _internal_set_landmark_thickness(from._internal_landmark_thickness());
  }
}

void IrisToRenderDataCalculatorOptions::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.IrisToRenderDataCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void IrisToRenderDataCalculatorOptions::CopyFrom(const IrisToRenderDataCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.IrisToRenderDataCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool IrisToRenderDataCalculatorOptions::IsInitialized() const {
  return true;
}

void IrisToRenderDataCalculatorOptions::InternalSwap(IrisToRenderDataCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(IrisToRenderDataCalculatorOptions, vertical_offset_px_)
      + sizeof(IrisToRenderDataCalculatorOptions::vertical_offset_px_)
      - PROTOBUF_FIELD_OFFSET(IrisToRenderDataCalculatorOptions, oval_color_)>(
          reinterpret_cast<char*>(&oval_color_),
          reinterpret_cast<char*>(&other->oval_color_));
  swap(font_height_px_, other->font_height_px_);
  swap(oval_thickness_, other->oval_thickness_);
  swap(landmark_thickness_, other->landmark_thickness_);
}

::PROTOBUF_NAMESPACE_ID::Metadata IrisToRenderDataCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fgraphs_2firis_5ftracking_2fcalculators_2firis_5fto_5frender_5fdata_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int IrisToRenderDataCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::IrisToRenderDataCalculatorOptions >, 11, false >
  IrisToRenderDataCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::IrisToRenderDataCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::IrisToRenderDataCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::IrisToRenderDataCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::IrisToRenderDataCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <x/google/protobuf/port_undef.inc>
