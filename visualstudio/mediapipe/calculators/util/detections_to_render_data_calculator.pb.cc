// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/util/detections_to_render_data_calculator.proto

#include "mediapipe/calculators/util/detections_to_render_data_calculator.pb.h"

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
constexpr DetectionsToRenderDataCalculatorOptions::DetectionsToRenderDataCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : text_delimiter_(nullptr)
  , scene_class_(nullptr)
  , text_(nullptr)
  , color_(nullptr)
  , one_label_per_line_(false)
  , render_detection_id_(false)
  , produce_empty_packet_(true)
  , thickness_(1){}
struct DetectionsToRenderDataCalculatorOptionsDefaultTypeInternal {
  constexpr DetectionsToRenderDataCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~DetectionsToRenderDataCalculatorOptionsDefaultTypeInternal() {}
  union {
    DetectionsToRenderDataCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT DetectionsToRenderDataCalculatorOptionsDefaultTypeInternal _DetectionsToRenderDataCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionsToRenderDataCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionsToRenderDataCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionsToRenderDataCalculatorOptions, produce_empty_packet_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionsToRenderDataCalculatorOptions, text_delimiter_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionsToRenderDataCalculatorOptions, one_label_per_line_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionsToRenderDataCalculatorOptions, text_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionsToRenderDataCalculatorOptions, thickness_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionsToRenderDataCalculatorOptions, color_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionsToRenderDataCalculatorOptions, scene_class_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionsToRenderDataCalculatorOptions, render_detection_id_),
  6,
  0,
  4,
  2,
  7,
  3,
  1,
  5,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 13, sizeof(::mediapipe::DetectionsToRenderDataCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_DetectionsToRenderDataCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\nEmediapipe/calculators/util/detections_"
  "to_render_data_calculator.proto\022\tmediapi"
  "pe\032$mediapipe/framework/calculator.proto"
  "\032\032mediapipe/util/color.proto\032 mediapipe/"
  "util/render_data.proto\"\230\003\n\'DetectionsToR"
  "enderDataCalculatorOptions\022\"\n\024produce_em"
  "pty_packet\030\001 \001(\010:\004true\022\031\n\016text_delimiter"
  "\030\002 \001(\t:\001,\022!\n\022one_label_per_line\030\003 \001(\010:\005f"
  "alse\022.\n\004text\030\004 \001(\0132 .mediapipe.RenderAnn"
  "otation.Text\022\024\n\tthickness\030\005 \001(\001:\0011\022\037\n\005co"
  "lor\030\006 \001(\0132\020.mediapipe.Color\022\036\n\013scene_cla"
  "ss\030\007 \001(\t:\tDETECTION\022\"\n\023render_detection_"
  "id\030\010 \001(\010:\005false2`\n\003ext\022\034.mediapipe.Calcu"
  "latorOptions\030\346\336\266v \001(\01322.mediapipe.Detect"
  "ionsToRenderDataCalculatorOptions"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto_deps[3] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
  &::descriptor_table_mediapipe_2futil_2fcolor_2eproto,
  &::descriptor_table_mediapipe_2futil_2frender_5fdata_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto = {
  false, false, 593, descriptor_table_protodef_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto, "mediapipe/calculators/util/detections_to_render_data_calculator.proto", 
  &descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto_deps, 3, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto(&descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto);
namespace mediapipe {

// ===================================================================

class DetectionsToRenderDataCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<DetectionsToRenderDataCalculatorOptions>()._has_bits_);
  static void set_has_produce_empty_packet(HasBits* has_bits) {
    (*has_bits)[0] |= 64u;
  }
  static void set_has_text_delimiter(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_one_label_per_line(HasBits* has_bits) {
    (*has_bits)[0] |= 16u;
  }
  static const ::mediapipe::RenderAnnotation_Text& text(const DetectionsToRenderDataCalculatorOptions* msg);
  static void set_has_text(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static void set_has_thickness(HasBits* has_bits) {
    (*has_bits)[0] |= 128u;
  }
  static const ::mediapipe::Color& color(const DetectionsToRenderDataCalculatorOptions* msg);
  static void set_has_color(HasBits* has_bits) {
    (*has_bits)[0] |= 8u;
  }
  static void set_has_scene_class(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static void set_has_render_detection_id(HasBits* has_bits) {
    (*has_bits)[0] |= 32u;
  }
};

const ::mediapipe::RenderAnnotation_Text&
DetectionsToRenderDataCalculatorOptions::_Internal::text(const DetectionsToRenderDataCalculatorOptions* msg) {
  return *msg->text_;
}
const ::mediapipe::Color&
DetectionsToRenderDataCalculatorOptions::_Internal::color(const DetectionsToRenderDataCalculatorOptions* msg) {
  return *msg->color_;
}
const ::PROTOBUF_NAMESPACE_ID::internal::LazyString DetectionsToRenderDataCalculatorOptions::_i_give_permission_to_break_this_code_default_text_delimiter_{{{",", 1}}, {nullptr}};
void DetectionsToRenderDataCalculatorOptions::clear_text() {
  if (text_ != nullptr) text_->Clear();
  _has_bits_[0] &= ~0x00000004u;
}
void DetectionsToRenderDataCalculatorOptions::clear_color() {
  if (color_ != nullptr) color_->Clear();
  _has_bits_[0] &= ~0x00000008u;
}
const ::PROTOBUF_NAMESPACE_ID::internal::LazyString DetectionsToRenderDataCalculatorOptions::_i_give_permission_to_break_this_code_default_scene_class_{{{"DETECTION", 9}}, {nullptr}};
DetectionsToRenderDataCalculatorOptions::DetectionsToRenderDataCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.DetectionsToRenderDataCalculatorOptions)
}
DetectionsToRenderDataCalculatorOptions::DetectionsToRenderDataCalculatorOptions(const DetectionsToRenderDataCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  text_delimiter_.UnsafeSetDefault(nullptr);
  if (from._internal_has_text_delimiter()) {
    text_delimiter_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::NonEmptyDefault{}, from._internal_text_delimiter(), 
      GetArena());
  }
  scene_class_.UnsafeSetDefault(nullptr);
  if (from._internal_has_scene_class()) {
    scene_class_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::NonEmptyDefault{}, from._internal_scene_class(), 
      GetArena());
  }
  if (from._internal_has_text()) {
    text_ = new ::mediapipe::RenderAnnotation_Text(*from.text_);
  } else {
    text_ = nullptr;
  }
  if (from._internal_has_color()) {
    color_ = new ::mediapipe::Color(*from.color_);
  } else {
    color_ = nullptr;
  }
  ::memcpy(&one_label_per_line_, &from.one_label_per_line_,
    static_cast<size_t>(reinterpret_cast<char*>(&thickness_) -
    reinterpret_cast<char*>(&one_label_per_line_)) + sizeof(thickness_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.DetectionsToRenderDataCalculatorOptions)
}

void DetectionsToRenderDataCalculatorOptions::SharedCtor() {
text_delimiter_.UnsafeSetDefault(nullptr);
scene_class_.UnsafeSetDefault(nullptr);
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&text_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&render_detection_id_) -
    reinterpret_cast<char*>(&text_)) + sizeof(render_detection_id_));
produce_empty_packet_ = true;
thickness_ = 1;
}

DetectionsToRenderDataCalculatorOptions::~DetectionsToRenderDataCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.DetectionsToRenderDataCalculatorOptions)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void DetectionsToRenderDataCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
  text_delimiter_.DestroyNoArena(nullptr);
  scene_class_.DestroyNoArena(nullptr);
  if (this != internal_default_instance()) delete text_;
  if (this != internal_default_instance()) delete color_;
}

void DetectionsToRenderDataCalculatorOptions::ArenaDtor(void* object) {
  DetectionsToRenderDataCalculatorOptions* _this = reinterpret_cast< DetectionsToRenderDataCalculatorOptions* >(object);
  (void)_this;
}
void DetectionsToRenderDataCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void DetectionsToRenderDataCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void DetectionsToRenderDataCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.DetectionsToRenderDataCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x0000000fu) {
    if (cached_has_bits & 0x00000001u) {
      text_delimiter_.ClearToDefault(::mediapipe::DetectionsToRenderDataCalculatorOptions::_i_give_permission_to_break_this_code_default_text_delimiter_, GetArena());
       }
    if (cached_has_bits & 0x00000002u) {
      scene_class_.ClearToDefault(::mediapipe::DetectionsToRenderDataCalculatorOptions::_i_give_permission_to_break_this_code_default_scene_class_, GetArena());
       }
    if (cached_has_bits & 0x00000004u) {
      GOOGLE_DCHECK(text_ != nullptr);
      text_->Clear();
    }
    if (cached_has_bits & 0x00000008u) {
      GOOGLE_DCHECK(color_ != nullptr);
      color_->Clear();
    }
  }
  ::memset(&one_label_per_line_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&render_detection_id_) -
      reinterpret_cast<char*>(&one_label_per_line_)) + sizeof(render_detection_id_));
  if (cached_has_bits & 0x000000c0u) {
    produce_empty_packet_ = true;
    thickness_ = 1;
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* DetectionsToRenderDataCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional bool produce_empty_packet = 1 [default = true];
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          _Internal::set_has_produce_empty_packet(&has_bits);
          produce_empty_packet_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional string text_delimiter = 2 [default = ","];
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          auto str = _internal_mutable_text_delimiter();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          #ifndef NDEBUG
          ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "mediapipe.DetectionsToRenderDataCalculatorOptions.text_delimiter");
          #endif  // !NDEBUG
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional bool one_label_per_line = 3 [default = false];
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          _Internal::set_has_one_label_per_line(&has_bits);
          one_label_per_line_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional .mediapipe.RenderAnnotation.Text text = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 34)) {
          ptr = ctx->ParseMessage(_internal_mutable_text(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional double thickness = 5 [default = 1];
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 41)) {
          _Internal::set_has_thickness(&has_bits);
          thickness_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr);
          ptr += sizeof(double);
        } else goto handle_unusual;
        continue;
      // optional .mediapipe.Color color = 6;
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 50)) {
          ptr = ctx->ParseMessage(_internal_mutable_color(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional string scene_class = 7 [default = "DETECTION"];
      case 7:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 58)) {
          auto str = _internal_mutable_scene_class();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          #ifndef NDEBUG
          ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "mediapipe.DetectionsToRenderDataCalculatorOptions.scene_class");
          #endif  // !NDEBUG
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional bool render_detection_id = 8 [default = false];
      case 8:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 64)) {
          _Internal::set_has_render_detection_id(&has_bits);
          render_detection_id_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
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

::PROTOBUF_NAMESPACE_ID::uint8* DetectionsToRenderDataCalculatorOptions::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.DetectionsToRenderDataCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional bool produce_empty_packet = 1 [default = true];
  if (cached_has_bits & 0x00000040u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(1, this->_internal_produce_empty_packet(), target);
  }

  // optional string text_delimiter = 2 [default = ","];
  if (cached_has_bits & 0x00000001u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->_internal_text_delimiter().data(), static_cast<int>(this->_internal_text_delimiter().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "mediapipe.DetectionsToRenderDataCalculatorOptions.text_delimiter");
    target = stream->WriteStringMaybeAliased(
        2, this->_internal_text_delimiter(), target);
  }

  // optional bool one_label_per_line = 3 [default = false];
  if (cached_has_bits & 0x00000010u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(3, this->_internal_one_label_per_line(), target);
  }

  // optional .mediapipe.RenderAnnotation.Text text = 4;
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        4, _Internal::text(this), target, stream);
  }

  // optional double thickness = 5 [default = 1];
  if (cached_has_bits & 0x00000080u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteDoubleToArray(5, this->_internal_thickness(), target);
  }

  // optional .mediapipe.Color color = 6;
  if (cached_has_bits & 0x00000008u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        6, _Internal::color(this), target, stream);
  }

  // optional string scene_class = 7 [default = "DETECTION"];
  if (cached_has_bits & 0x00000002u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->_internal_scene_class().data(), static_cast<int>(this->_internal_scene_class().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "mediapipe.DetectionsToRenderDataCalculatorOptions.scene_class");
    target = stream->WriteStringMaybeAliased(
        7, this->_internal_scene_class(), target);
  }

  // optional bool render_detection_id = 8 [default = false];
  if (cached_has_bits & 0x00000020u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(8, this->_internal_render_detection_id(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.DetectionsToRenderDataCalculatorOptions)
  return target;
}

size_t DetectionsToRenderDataCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.DetectionsToRenderDataCalculatorOptions)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    // optional string text_delimiter = 2 [default = ","];
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_text_delimiter());
    }

    // optional string scene_class = 7 [default = "DETECTION"];
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_scene_class());
    }

    // optional .mediapipe.RenderAnnotation.Text text = 4;
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *text_);
    }

    // optional .mediapipe.Color color = 6;
    if (cached_has_bits & 0x00000008u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *color_);
    }

    // optional bool one_label_per_line = 3 [default = false];
    if (cached_has_bits & 0x00000010u) {
      total_size += 1 + 1;
    }

    // optional bool render_detection_id = 8 [default = false];
    if (cached_has_bits & 0x00000020u) {
      total_size += 1 + 1;
    }

    // optional bool produce_empty_packet = 1 [default = true];
    if (cached_has_bits & 0x00000040u) {
      total_size += 1 + 1;
    }

    // optional double thickness = 5 [default = 1];
    if (cached_has_bits & 0x00000080u) {
      total_size += 1 + 8;
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

void DetectionsToRenderDataCalculatorOptions::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.DetectionsToRenderDataCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  const DetectionsToRenderDataCalculatorOptions* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<DetectionsToRenderDataCalculatorOptions>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.DetectionsToRenderDataCalculatorOptions)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.DetectionsToRenderDataCalculatorOptions)
    MergeFrom(*source);
  }
}

void DetectionsToRenderDataCalculatorOptions::MergeFrom(const DetectionsToRenderDataCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.DetectionsToRenderDataCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    if (cached_has_bits & 0x00000001u) {
      _internal_set_text_delimiter(from._internal_text_delimiter());
    }
    if (cached_has_bits & 0x00000002u) {
      _internal_set_scene_class(from._internal_scene_class());
    }
    if (cached_has_bits & 0x00000004u) {
      _internal_mutable_text()->::mediapipe::RenderAnnotation_Text::MergeFrom(from._internal_text());
    }
    if (cached_has_bits & 0x00000008u) {
      _internal_mutable_color()->::mediapipe::Color::MergeFrom(from._internal_color());
    }
    if (cached_has_bits & 0x00000010u) {
      one_label_per_line_ = from.one_label_per_line_;
    }
    if (cached_has_bits & 0x00000020u) {
      render_detection_id_ = from.render_detection_id_;
    }
    if (cached_has_bits & 0x00000040u) {
      produce_empty_packet_ = from.produce_empty_packet_;
    }
    if (cached_has_bits & 0x00000080u) {
      thickness_ = from.thickness_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void DetectionsToRenderDataCalculatorOptions::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.DetectionsToRenderDataCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void DetectionsToRenderDataCalculatorOptions::CopyFrom(const DetectionsToRenderDataCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.DetectionsToRenderDataCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool DetectionsToRenderDataCalculatorOptions::IsInitialized() const {
  return true;
}

void DetectionsToRenderDataCalculatorOptions::InternalSwap(DetectionsToRenderDataCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  text_delimiter_.Swap(&other->text_delimiter_, nullptr, GetArena());
  scene_class_.Swap(&other->scene_class_, nullptr, GetArena());
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(DetectionsToRenderDataCalculatorOptions, render_detection_id_)
      + sizeof(DetectionsToRenderDataCalculatorOptions::render_detection_id_)
      - PROTOBUF_FIELD_OFFSET(DetectionsToRenderDataCalculatorOptions, text_)>(
          reinterpret_cast<char*>(&text_),
          reinterpret_cast<char*>(&other->text_));
  swap(produce_empty_packet_, other->produce_empty_packet_);
  swap(thickness_, other->thickness_);
}

::PROTOBUF_NAMESPACE_ID::Metadata DetectionsToRenderDataCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2futil_2fdetections_5fto_5frender_5fdata_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int DetectionsToRenderDataCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::DetectionsToRenderDataCalculatorOptions >, 11, false >
  DetectionsToRenderDataCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::DetectionsToRenderDataCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::DetectionsToRenderDataCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::DetectionsToRenderDataCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::DetectionsToRenderDataCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <x/google/protobuf/port_undef.inc>
