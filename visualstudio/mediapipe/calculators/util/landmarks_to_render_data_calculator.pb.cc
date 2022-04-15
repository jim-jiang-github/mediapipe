// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/util/landmarks_to_render_data_calculator.proto

#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"

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
constexpr LandmarksToRenderDataCalculatorOptions::LandmarksToRenderDataCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : landmark_connections_()
  , landmark_color_(nullptr)
  , connection_color_(nullptr)
  , min_depth_line_color_(nullptr)
  , max_depth_line_color_(nullptr)
  , visibility_threshold_(0)
  , presence_threshold_(0)
  , min_depth_circle_thickness_(0)
  , utilize_visibility_(false)
  , utilize_presence_(false)
  , visualize_landmark_depth_(true)
  , thickness_(1)
  , max_depth_circle_thickness_(18){}
struct LandmarksToRenderDataCalculatorOptionsDefaultTypeInternal {
  constexpr LandmarksToRenderDataCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~LandmarksToRenderDataCalculatorOptionsDefaultTypeInternal() {}
  union {
    LandmarksToRenderDataCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT LandmarksToRenderDataCalculatorOptionsDefaultTypeInternal _LandmarksToRenderDataCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::LandmarksToRenderDataCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::LandmarksToRenderDataCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::LandmarksToRenderDataCalculatorOptions, landmark_connections_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::LandmarksToRenderDataCalculatorOptions, landmark_color_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::LandmarksToRenderDataCalculatorOptions, connection_color_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::LandmarksToRenderDataCalculatorOptions, thickness_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::LandmarksToRenderDataCalculatorOptions, visualize_landmark_depth_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::LandmarksToRenderDataCalculatorOptions, utilize_visibility_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::LandmarksToRenderDataCalculatorOptions, visibility_threshold_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::LandmarksToRenderDataCalculatorOptions, utilize_presence_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::LandmarksToRenderDataCalculatorOptions, presence_threshold_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::LandmarksToRenderDataCalculatorOptions, min_depth_circle_thickness_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::LandmarksToRenderDataCalculatorOptions, max_depth_circle_thickness_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::LandmarksToRenderDataCalculatorOptions, min_depth_line_color_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::LandmarksToRenderDataCalculatorOptions, max_depth_line_color_),
  ~0u,
  0,
  1,
  10,
  9,
  7,
  4,
  8,
  5,
  6,
  11,
  2,
  3,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 18, sizeof(::mediapipe::LandmarksToRenderDataCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_LandmarksToRenderDataCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\nDmediapipe/calculators/util/landmarks_t"
  "o_render_data_calculator.proto\022\tmediapip"
  "e\032$mediapipe/framework/calculator.proto\032"
  "\032mediapipe/util/color.proto\"\356\004\n&Landmark"
  "sToRenderDataCalculatorOptions\022\034\n\024landma"
  "rk_connections\030\001 \003(\005\022(\n\016landmark_color\030\002"
  " \001(\0132\020.mediapipe.Color\022*\n\020connection_col"
  "or\030\003 \001(\0132\020.mediapipe.Color\022\024\n\tthickness\030"
  "\004 \001(\001:\0011\022&\n\030visualize_landmark_depth\030\005 \001"
  "(\010:\004true\022!\n\022utilize_visibility\030\006 \001(\010:\005fa"
  "lse\022\037\n\024visibility_threshold\030\007 \001(\001:\0010\022\037\n\020"
  "utilize_presence\030\010 \001(\010:\005false\022\035\n\022presenc"
  "e_threshold\030\t \001(\001:\0010\022%\n\032min_depth_circle"
  "_thickness\030\n \001(\001:\0010\022&\n\032max_depth_circle_"
  "thickness\030\013 \001(\001:\00218\022.\n\024min_depth_line_co"
  "lor\030\014 \001(\0132\020.mediapipe.Color\022.\n\024max_depth"
  "_line_color\030\r \001(\0132\020.mediapipe.Color2_\n\003e"
  "xt\022\034.mediapipe.CalculatorOptions\030\275\322\235{ \001("
  "\01321.mediapipe.LandmarksToRenderDataCalcu"
  "latorOptions"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto_deps[2] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
  &::descriptor_table_mediapipe_2futil_2fcolor_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto = {
  false, false, 772, descriptor_table_protodef_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto, "mediapipe/calculators/util/landmarks_to_render_data_calculator.proto", 
  &descriptor_table_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto_deps, 2, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto(&descriptor_table_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto);
namespace mediapipe {

// ===================================================================

class LandmarksToRenderDataCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<LandmarksToRenderDataCalculatorOptions>()._has_bits_);
  static const ::mediapipe::Color& landmark_color(const LandmarksToRenderDataCalculatorOptions* msg);
  static void set_has_landmark_color(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static const ::mediapipe::Color& connection_color(const LandmarksToRenderDataCalculatorOptions* msg);
  static void set_has_connection_color(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static void set_has_thickness(HasBits* has_bits) {
    (*has_bits)[0] |= 1024u;
  }
  static void set_has_visualize_landmark_depth(HasBits* has_bits) {
    (*has_bits)[0] |= 512u;
  }
  static void set_has_utilize_visibility(HasBits* has_bits) {
    (*has_bits)[0] |= 128u;
  }
  static void set_has_visibility_threshold(HasBits* has_bits) {
    (*has_bits)[0] |= 16u;
  }
  static void set_has_utilize_presence(HasBits* has_bits) {
    (*has_bits)[0] |= 256u;
  }
  static void set_has_presence_threshold(HasBits* has_bits) {
    (*has_bits)[0] |= 32u;
  }
  static void set_has_min_depth_circle_thickness(HasBits* has_bits) {
    (*has_bits)[0] |= 64u;
  }
  static void set_has_max_depth_circle_thickness(HasBits* has_bits) {
    (*has_bits)[0] |= 2048u;
  }
  static const ::mediapipe::Color& min_depth_line_color(const LandmarksToRenderDataCalculatorOptions* msg);
  static void set_has_min_depth_line_color(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static const ::mediapipe::Color& max_depth_line_color(const LandmarksToRenderDataCalculatorOptions* msg);
  static void set_has_max_depth_line_color(HasBits* has_bits) {
    (*has_bits)[0] |= 8u;
  }
};

const ::mediapipe::Color&
LandmarksToRenderDataCalculatorOptions::_Internal::landmark_color(const LandmarksToRenderDataCalculatorOptions* msg) {
  return *msg->landmark_color_;
}
const ::mediapipe::Color&
LandmarksToRenderDataCalculatorOptions::_Internal::connection_color(const LandmarksToRenderDataCalculatorOptions* msg) {
  return *msg->connection_color_;
}
const ::mediapipe::Color&
LandmarksToRenderDataCalculatorOptions::_Internal::min_depth_line_color(const LandmarksToRenderDataCalculatorOptions* msg) {
  return *msg->min_depth_line_color_;
}
const ::mediapipe::Color&
LandmarksToRenderDataCalculatorOptions::_Internal::max_depth_line_color(const LandmarksToRenderDataCalculatorOptions* msg) {
  return *msg->max_depth_line_color_;
}
void LandmarksToRenderDataCalculatorOptions::clear_landmark_color() {
  if (landmark_color_ != nullptr) landmark_color_->Clear();
  _has_bits_[0] &= ~0x00000001u;
}
void LandmarksToRenderDataCalculatorOptions::clear_connection_color() {
  if (connection_color_ != nullptr) connection_color_->Clear();
  _has_bits_[0] &= ~0x00000002u;
}
void LandmarksToRenderDataCalculatorOptions::clear_min_depth_line_color() {
  if (min_depth_line_color_ != nullptr) min_depth_line_color_->Clear();
  _has_bits_[0] &= ~0x00000004u;
}
void LandmarksToRenderDataCalculatorOptions::clear_max_depth_line_color() {
  if (max_depth_line_color_ != nullptr) max_depth_line_color_->Clear();
  _has_bits_[0] &= ~0x00000008u;
}
LandmarksToRenderDataCalculatorOptions::LandmarksToRenderDataCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena),
  landmark_connections_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.LandmarksToRenderDataCalculatorOptions)
}
LandmarksToRenderDataCalculatorOptions::LandmarksToRenderDataCalculatorOptions(const LandmarksToRenderDataCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_),
      landmark_connections_(from.landmark_connections_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  if (from._internal_has_landmark_color()) {
    landmark_color_ = new ::mediapipe::Color(*from.landmark_color_);
  } else {
    landmark_color_ = nullptr;
  }
  if (from._internal_has_connection_color()) {
    connection_color_ = new ::mediapipe::Color(*from.connection_color_);
  } else {
    connection_color_ = nullptr;
  }
  if (from._internal_has_min_depth_line_color()) {
    min_depth_line_color_ = new ::mediapipe::Color(*from.min_depth_line_color_);
  } else {
    min_depth_line_color_ = nullptr;
  }
  if (from._internal_has_max_depth_line_color()) {
    max_depth_line_color_ = new ::mediapipe::Color(*from.max_depth_line_color_);
  } else {
    max_depth_line_color_ = nullptr;
  }
  ::memcpy(&visibility_threshold_, &from.visibility_threshold_,
    static_cast<size_t>(reinterpret_cast<char*>(&max_depth_circle_thickness_) -
    reinterpret_cast<char*>(&visibility_threshold_)) + sizeof(max_depth_circle_thickness_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.LandmarksToRenderDataCalculatorOptions)
}

void LandmarksToRenderDataCalculatorOptions::SharedCtor() {
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&landmark_color_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&utilize_presence_) -
    reinterpret_cast<char*>(&landmark_color_)) + sizeof(utilize_presence_));
visualize_landmark_depth_ = true;
thickness_ = 1;
max_depth_circle_thickness_ = 18;
}

LandmarksToRenderDataCalculatorOptions::~LandmarksToRenderDataCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.LandmarksToRenderDataCalculatorOptions)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void LandmarksToRenderDataCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
  if (this != internal_default_instance()) delete landmark_color_;
  if (this != internal_default_instance()) delete connection_color_;
  if (this != internal_default_instance()) delete min_depth_line_color_;
  if (this != internal_default_instance()) delete max_depth_line_color_;
}

void LandmarksToRenderDataCalculatorOptions::ArenaDtor(void* object) {
  LandmarksToRenderDataCalculatorOptions* _this = reinterpret_cast< LandmarksToRenderDataCalculatorOptions* >(object);
  (void)_this;
}
void LandmarksToRenderDataCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void LandmarksToRenderDataCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void LandmarksToRenderDataCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.LandmarksToRenderDataCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  landmark_connections_.Clear();
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x0000000fu) {
    if (cached_has_bits & 0x00000001u) {
      GOOGLE_DCHECK(landmark_color_ != nullptr);
      landmark_color_->Clear();
    }
    if (cached_has_bits & 0x00000002u) {
      GOOGLE_DCHECK(connection_color_ != nullptr);
      connection_color_->Clear();
    }
    if (cached_has_bits & 0x00000004u) {
      GOOGLE_DCHECK(min_depth_line_color_ != nullptr);
      min_depth_line_color_->Clear();
    }
    if (cached_has_bits & 0x00000008u) {
      GOOGLE_DCHECK(max_depth_line_color_ != nullptr);
      max_depth_line_color_->Clear();
    }
  }
  if (cached_has_bits & 0x000000f0u) {
    ::memset(&visibility_threshold_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&utilize_visibility_) -
        reinterpret_cast<char*>(&visibility_threshold_)) + sizeof(utilize_visibility_));
  }
  if (cached_has_bits & 0x00000f00u) {
    utilize_presence_ = false;
    visualize_landmark_depth_ = true;
    thickness_ = 1;
    max_depth_circle_thickness_ = 18;
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* LandmarksToRenderDataCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // repeated int32 landmark_connections = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          ptr -= 1;
          do {
            ptr += 1;
            _internal_add_landmark_connections(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr));
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<8>(ptr));
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt32Parser(_internal_mutable_landmark_connections(), ptr, ctx);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional .mediapipe.Color landmark_color = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          ptr = ctx->ParseMessage(_internal_mutable_landmark_color(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional .mediapipe.Color connection_color = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 26)) {
          ptr = ctx->ParseMessage(_internal_mutable_connection_color(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional double thickness = 4 [default = 1];
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 33)) {
          _Internal::set_has_thickness(&has_bits);
          thickness_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr);
          ptr += sizeof(double);
        } else goto handle_unusual;
        continue;
      // optional bool visualize_landmark_depth = 5 [default = true];
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 40)) {
          _Internal::set_has_visualize_landmark_depth(&has_bits);
          visualize_landmark_depth_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional bool utilize_visibility = 6 [default = false];
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 48)) {
          _Internal::set_has_utilize_visibility(&has_bits);
          utilize_visibility_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional double visibility_threshold = 7 [default = 0];
      case 7:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 57)) {
          _Internal::set_has_visibility_threshold(&has_bits);
          visibility_threshold_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr);
          ptr += sizeof(double);
        } else goto handle_unusual;
        continue;
      // optional bool utilize_presence = 8 [default = false];
      case 8:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 64)) {
          _Internal::set_has_utilize_presence(&has_bits);
          utilize_presence_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional double presence_threshold = 9 [default = 0];
      case 9:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 73)) {
          _Internal::set_has_presence_threshold(&has_bits);
          presence_threshold_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr);
          ptr += sizeof(double);
        } else goto handle_unusual;
        continue;
      // optional double min_depth_circle_thickness = 10 [default = 0];
      case 10:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 81)) {
          _Internal::set_has_min_depth_circle_thickness(&has_bits);
          min_depth_circle_thickness_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr);
          ptr += sizeof(double);
        } else goto handle_unusual;
        continue;
      // optional double max_depth_circle_thickness = 11 [default = 18];
      case 11:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 89)) {
          _Internal::set_has_max_depth_circle_thickness(&has_bits);
          max_depth_circle_thickness_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<double>(ptr);
          ptr += sizeof(double);
        } else goto handle_unusual;
        continue;
      // optional .mediapipe.Color min_depth_line_color = 12;
      case 12:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 98)) {
          ptr = ctx->ParseMessage(_internal_mutable_min_depth_line_color(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional .mediapipe.Color max_depth_line_color = 13;
      case 13:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 106)) {
          ptr = ctx->ParseMessage(_internal_mutable_max_depth_line_color(), ptr);
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

::PROTOBUF_NAMESPACE_ID::uint8* LandmarksToRenderDataCalculatorOptions::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.LandmarksToRenderDataCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated int32 landmark_connections = 1;
  for (int i = 0, n = this->_internal_landmark_connections_size(); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->_internal_landmark_connections(i), target);
  }

  cached_has_bits = _has_bits_[0];
  // optional .mediapipe.Color landmark_color = 2;
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        2, _Internal::landmark_color(this), target, stream);
  }

  // optional .mediapipe.Color connection_color = 3;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        3, _Internal::connection_color(this), target, stream);
  }

  // optional double thickness = 4 [default = 1];
  if (cached_has_bits & 0x00000400u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteDoubleToArray(4, this->_internal_thickness(), target);
  }

  // optional bool visualize_landmark_depth = 5 [default = true];
  if (cached_has_bits & 0x00000200u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(5, this->_internal_visualize_landmark_depth(), target);
  }

  // optional bool utilize_visibility = 6 [default = false];
  if (cached_has_bits & 0x00000080u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(6, this->_internal_utilize_visibility(), target);
  }

  // optional double visibility_threshold = 7 [default = 0];
  if (cached_has_bits & 0x00000010u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteDoubleToArray(7, this->_internal_visibility_threshold(), target);
  }

  // optional bool utilize_presence = 8 [default = false];
  if (cached_has_bits & 0x00000100u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(8, this->_internal_utilize_presence(), target);
  }

  // optional double presence_threshold = 9 [default = 0];
  if (cached_has_bits & 0x00000020u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteDoubleToArray(9, this->_internal_presence_threshold(), target);
  }

  // optional double min_depth_circle_thickness = 10 [default = 0];
  if (cached_has_bits & 0x00000040u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteDoubleToArray(10, this->_internal_min_depth_circle_thickness(), target);
  }

  // optional double max_depth_circle_thickness = 11 [default = 18];
  if (cached_has_bits & 0x00000800u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteDoubleToArray(11, this->_internal_max_depth_circle_thickness(), target);
  }

  // optional .mediapipe.Color min_depth_line_color = 12;
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        12, _Internal::min_depth_line_color(this), target, stream);
  }

  // optional .mediapipe.Color max_depth_line_color = 13;
  if (cached_has_bits & 0x00000008u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        13, _Internal::max_depth_line_color(this), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.LandmarksToRenderDataCalculatorOptions)
  return target;
}

size_t LandmarksToRenderDataCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.LandmarksToRenderDataCalculatorOptions)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated int32 landmark_connections = 1;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      Int32Size(this->landmark_connections_);
    total_size += 1 *
                  ::PROTOBUF_NAMESPACE_ID::internal::FromIntSize(this->_internal_landmark_connections_size());
    total_size += data_size;
  }

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    // optional .mediapipe.Color landmark_color = 2;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *landmark_color_);
    }

    // optional .mediapipe.Color connection_color = 3;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *connection_color_);
    }

    // optional .mediapipe.Color min_depth_line_color = 12;
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *min_depth_line_color_);
    }

    // optional .mediapipe.Color max_depth_line_color = 13;
    if (cached_has_bits & 0x00000008u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *max_depth_line_color_);
    }

    // optional double visibility_threshold = 7 [default = 0];
    if (cached_has_bits & 0x00000010u) {
      total_size += 1 + 8;
    }

    // optional double presence_threshold = 9 [default = 0];
    if (cached_has_bits & 0x00000020u) {
      total_size += 1 + 8;
    }

    // optional double min_depth_circle_thickness = 10 [default = 0];
    if (cached_has_bits & 0x00000040u) {
      total_size += 1 + 8;
    }

    // optional bool utilize_visibility = 6 [default = false];
    if (cached_has_bits & 0x00000080u) {
      total_size += 1 + 1;
    }

  }
  if (cached_has_bits & 0x00000f00u) {
    // optional bool utilize_presence = 8 [default = false];
    if (cached_has_bits & 0x00000100u) {
      total_size += 1 + 1;
    }

    // optional bool visualize_landmark_depth = 5 [default = true];
    if (cached_has_bits & 0x00000200u) {
      total_size += 1 + 1;
    }

    // optional double thickness = 4 [default = 1];
    if (cached_has_bits & 0x00000400u) {
      total_size += 1 + 8;
    }

    // optional double max_depth_circle_thickness = 11 [default = 18];
    if (cached_has_bits & 0x00000800u) {
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

void LandmarksToRenderDataCalculatorOptions::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.LandmarksToRenderDataCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  const LandmarksToRenderDataCalculatorOptions* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<LandmarksToRenderDataCalculatorOptions>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.LandmarksToRenderDataCalculatorOptions)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.LandmarksToRenderDataCalculatorOptions)
    MergeFrom(*source);
  }
}

void LandmarksToRenderDataCalculatorOptions::MergeFrom(const LandmarksToRenderDataCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.LandmarksToRenderDataCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  landmark_connections_.MergeFrom(from.landmark_connections_);
  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    if (cached_has_bits & 0x00000001u) {
      _internal_mutable_landmark_color()->::mediapipe::Color::MergeFrom(from._internal_landmark_color());
    }
    if (cached_has_bits & 0x00000002u) {
      _internal_mutable_connection_color()->::mediapipe::Color::MergeFrom(from._internal_connection_color());
    }
    if (cached_has_bits & 0x00000004u) {
      _internal_mutable_min_depth_line_color()->::mediapipe::Color::MergeFrom(from._internal_min_depth_line_color());
    }
    if (cached_has_bits & 0x00000008u) {
      _internal_mutable_max_depth_line_color()->::mediapipe::Color::MergeFrom(from._internal_max_depth_line_color());
    }
    if (cached_has_bits & 0x00000010u) {
      visibility_threshold_ = from.visibility_threshold_;
    }
    if (cached_has_bits & 0x00000020u) {
      presence_threshold_ = from.presence_threshold_;
    }
    if (cached_has_bits & 0x00000040u) {
      min_depth_circle_thickness_ = from.min_depth_circle_thickness_;
    }
    if (cached_has_bits & 0x00000080u) {
      utilize_visibility_ = from.utilize_visibility_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
  if (cached_has_bits & 0x00000f00u) {
    if (cached_has_bits & 0x00000100u) {
      utilize_presence_ = from.utilize_presence_;
    }
    if (cached_has_bits & 0x00000200u) {
      visualize_landmark_depth_ = from.visualize_landmark_depth_;
    }
    if (cached_has_bits & 0x00000400u) {
      thickness_ = from.thickness_;
    }
    if (cached_has_bits & 0x00000800u) {
      max_depth_circle_thickness_ = from.max_depth_circle_thickness_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void LandmarksToRenderDataCalculatorOptions::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.LandmarksToRenderDataCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void LandmarksToRenderDataCalculatorOptions::CopyFrom(const LandmarksToRenderDataCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.LandmarksToRenderDataCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool LandmarksToRenderDataCalculatorOptions::IsInitialized() const {
  return true;
}

void LandmarksToRenderDataCalculatorOptions::InternalSwap(LandmarksToRenderDataCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  landmark_connections_.InternalSwap(&other->landmark_connections_);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(LandmarksToRenderDataCalculatorOptions, utilize_presence_)
      + sizeof(LandmarksToRenderDataCalculatorOptions::utilize_presence_)
      - PROTOBUF_FIELD_OFFSET(LandmarksToRenderDataCalculatorOptions, landmark_color_)>(
          reinterpret_cast<char*>(&landmark_color_),
          reinterpret_cast<char*>(&other->landmark_color_));
  swap(visualize_landmark_depth_, other->visualize_landmark_depth_);
  swap(thickness_, other->thickness_);
  swap(max_depth_circle_thickness_, other->max_depth_circle_thickness_);
}

::PROTOBUF_NAMESPACE_ID::Metadata LandmarksToRenderDataCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2futil_2flandmarks_5fto_5frender_5fdata_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int LandmarksToRenderDataCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::LandmarksToRenderDataCalculatorOptions >, 11, false >
  LandmarksToRenderDataCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::LandmarksToRenderDataCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::LandmarksToRenderDataCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::LandmarksToRenderDataCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::LandmarksToRenderDataCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
