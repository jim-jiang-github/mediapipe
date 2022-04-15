// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto

#include "mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.pb.h"

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
constexpr TfLiteTensorsToDetectionsCalculatorOptions::TfLiteTensorsToDetectionsCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : ignore_classes_()
  , num_classes_(0)
  , num_boxes_(0)
  , num_coords_(0)
  , x_scale_(0)
  , y_scale_(0)
  , w_scale_(0)
  , h_scale_(0)
  , keypoint_coord_offset_(0)
  , num_keypoints_(0)
  , box_coord_offset_(0)
  , apply_exponential_on_box_size_(false)
  , reverse_output_order_(false)
  , sigmoid_score_(false)
  , flip_vertically_(false)
  , score_clipping_thresh_(0)
  , min_score_thresh_(0)
  , num_values_per_keypoint_(2){}
struct TfLiteTensorsToDetectionsCalculatorOptionsDefaultTypeInternal {
  constexpr TfLiteTensorsToDetectionsCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~TfLiteTensorsToDetectionsCalculatorOptionsDefaultTypeInternal() {}
  union {
    TfLiteTensorsToDetectionsCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT TfLiteTensorsToDetectionsCalculatorOptionsDefaultTypeInternal _TfLiteTensorsToDetectionsCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, num_classes_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, num_boxes_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, num_coords_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, keypoint_coord_offset_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, num_keypoints_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, num_values_per_keypoint_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, box_coord_offset_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, x_scale_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, y_scale_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, w_scale_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, h_scale_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, apply_exponential_on_box_size_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, reverse_output_order_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, ignore_classes_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, sigmoid_score_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, score_clipping_thresh_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, flip_vertically_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions, min_score_thresh_),
  0,
  1,
  2,
  7,
  8,
  16,
  9,
  3,
  4,
  5,
  6,
  10,
  11,
  ~0u,
  12,
  14,
  13,
  15,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 23, sizeof(::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_TfLiteTensorsToDetectionsCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\nJmediapipe/calculators/tflite/tflite_te"
  "nsors_to_detections_calculator.proto\022\tme"
  "diapipe\032$mediapipe/framework/calculator."
  "proto\"\371\004\n*TfLiteTensorsToDetectionsCalcu"
  "latorOptions\022\023\n\013num_classes\030\001 \001(\005\022\021\n\tnum"
  "_boxes\030\002 \001(\005\022\022\n\nnum_coords\030\003 \001(\005\022\035\n\025keyp"
  "oint_coord_offset\030\t \001(\005\022\030\n\rnum_keypoints"
  "\030\n \001(\005:\0010\022\"\n\027num_values_per_keypoint\030\013 \001"
  "(\005:\0012\022\033\n\020box_coord_offset\030\014 \001(\005:\0010\022\022\n\007x_"
  "scale\030\004 \001(\002:\0010\022\022\n\007y_scale\030\005 \001(\002:\0010\022\022\n\007w_"
  "scale\030\006 \001(\002:\0010\022\022\n\007h_scale\030\007 \001(\002:\0010\022,\n\035ap"
  "ply_exponential_on_box_size\030\r \001(\010:\005false"
  "\022#\n\024reverse_output_order\030\016 \001(\010:\005false\022\026\n"
  "\016ignore_classes\030\010 \003(\005\022\034\n\rsigmoid_score\030\017"
  " \001(\010:\005false\022\035\n\025score_clipping_thresh\030\020 \001"
  "(\002\022\036\n\017flip_vertically\030\022 \001(\010:\005false\022\030\n\020mi"
  "n_score_thresh\030\023 \001(\0022c\n\003ext\022\034.mediapipe."
  "CalculatorOptions\030\230\212\306u \001(\01325.mediapipe.T"
  "fLiteTensorsToDetectionsCalculatorOption"
  "s"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto_deps[1] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto = {
  false, false, 761, descriptor_table_protodef_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto, "mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto", 
  &descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto_deps, 1, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto(&descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto);
namespace mediapipe {

// ===================================================================

class TfLiteTensorsToDetectionsCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<TfLiteTensorsToDetectionsCalculatorOptions>()._has_bits_);
  static void set_has_num_classes(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_num_boxes(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static void set_has_num_coords(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static void set_has_keypoint_coord_offset(HasBits* has_bits) {
    (*has_bits)[0] |= 128u;
  }
  static void set_has_num_keypoints(HasBits* has_bits) {
    (*has_bits)[0] |= 256u;
  }
  static void set_has_num_values_per_keypoint(HasBits* has_bits) {
    (*has_bits)[0] |= 65536u;
  }
  static void set_has_box_coord_offset(HasBits* has_bits) {
    (*has_bits)[0] |= 512u;
  }
  static void set_has_x_scale(HasBits* has_bits) {
    (*has_bits)[0] |= 8u;
  }
  static void set_has_y_scale(HasBits* has_bits) {
    (*has_bits)[0] |= 16u;
  }
  static void set_has_w_scale(HasBits* has_bits) {
    (*has_bits)[0] |= 32u;
  }
  static void set_has_h_scale(HasBits* has_bits) {
    (*has_bits)[0] |= 64u;
  }
  static void set_has_apply_exponential_on_box_size(HasBits* has_bits) {
    (*has_bits)[0] |= 1024u;
  }
  static void set_has_reverse_output_order(HasBits* has_bits) {
    (*has_bits)[0] |= 2048u;
  }
  static void set_has_sigmoid_score(HasBits* has_bits) {
    (*has_bits)[0] |= 4096u;
  }
  static void set_has_score_clipping_thresh(HasBits* has_bits) {
    (*has_bits)[0] |= 16384u;
  }
  static void set_has_flip_vertically(HasBits* has_bits) {
    (*has_bits)[0] |= 8192u;
  }
  static void set_has_min_score_thresh(HasBits* has_bits) {
    (*has_bits)[0] |= 32768u;
  }
};

TfLiteTensorsToDetectionsCalculatorOptions::TfLiteTensorsToDetectionsCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena),
  ignore_classes_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.TfLiteTensorsToDetectionsCalculatorOptions)
}
TfLiteTensorsToDetectionsCalculatorOptions::TfLiteTensorsToDetectionsCalculatorOptions(const TfLiteTensorsToDetectionsCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_),
      ignore_classes_(from.ignore_classes_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&num_classes_, &from.num_classes_,
    static_cast<size_t>(reinterpret_cast<char*>(&num_values_per_keypoint_) -
    reinterpret_cast<char*>(&num_classes_)) + sizeof(num_values_per_keypoint_));
  // @@protoc_insertion_point(copy_constructor:mediapipe.TfLiteTensorsToDetectionsCalculatorOptions)
}

void TfLiteTensorsToDetectionsCalculatorOptions::SharedCtor() {
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&num_classes_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&min_score_thresh_) -
    reinterpret_cast<char*>(&num_classes_)) + sizeof(min_score_thresh_));
num_values_per_keypoint_ = 2;
}

TfLiteTensorsToDetectionsCalculatorOptions::~TfLiteTensorsToDetectionsCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.TfLiteTensorsToDetectionsCalculatorOptions)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void TfLiteTensorsToDetectionsCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void TfLiteTensorsToDetectionsCalculatorOptions::ArenaDtor(void* object) {
  TfLiteTensorsToDetectionsCalculatorOptions* _this = reinterpret_cast< TfLiteTensorsToDetectionsCalculatorOptions* >(object);
  (void)_this;
}
void TfLiteTensorsToDetectionsCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void TfLiteTensorsToDetectionsCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void TfLiteTensorsToDetectionsCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.TfLiteTensorsToDetectionsCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  ignore_classes_.Clear();
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    ::memset(&num_classes_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&keypoint_coord_offset_) -
        reinterpret_cast<char*>(&num_classes_)) + sizeof(keypoint_coord_offset_));
  }
  if (cached_has_bits & 0x0000ff00u) {
    ::memset(&num_keypoints_, 0, static_cast<size_t>(
        reinterpret_cast<char*>(&min_score_thresh_) -
        reinterpret_cast<char*>(&num_keypoints_)) + sizeof(min_score_thresh_));
  }
  num_values_per_keypoint_ = 2;
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* TfLiteTensorsToDetectionsCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional int32 num_classes = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          _Internal::set_has_num_classes(&has_bits);
          num_classes_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 num_boxes = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          _Internal::set_has_num_boxes(&has_bits);
          num_boxes_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 num_coords = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          _Internal::set_has_num_coords(&has_bits);
          num_coords_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional float x_scale = 4 [default = 0];
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 37)) {
          _Internal::set_has_x_scale(&has_bits);
          x_scale_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else goto handle_unusual;
        continue;
      // optional float y_scale = 5 [default = 0];
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 45)) {
          _Internal::set_has_y_scale(&has_bits);
          y_scale_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else goto handle_unusual;
        continue;
      // optional float w_scale = 6 [default = 0];
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 53)) {
          _Internal::set_has_w_scale(&has_bits);
          w_scale_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else goto handle_unusual;
        continue;
      // optional float h_scale = 7 [default = 0];
      case 7:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 61)) {
          _Internal::set_has_h_scale(&has_bits);
          h_scale_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else goto handle_unusual;
        continue;
      // repeated int32 ignore_classes = 8;
      case 8:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 64)) {
          ptr -= 1;
          do {
            ptr += 1;
            _internal_add_ignore_classes(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr));
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<64>(ptr));
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 66) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt32Parser(_internal_mutable_ignore_classes(), ptr, ctx);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 keypoint_coord_offset = 9;
      case 9:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 72)) {
          _Internal::set_has_keypoint_coord_offset(&has_bits);
          keypoint_coord_offset_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 num_keypoints = 10 [default = 0];
      case 10:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 80)) {
          _Internal::set_has_num_keypoints(&has_bits);
          num_keypoints_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 num_values_per_keypoint = 11 [default = 2];
      case 11:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 88)) {
          _Internal::set_has_num_values_per_keypoint(&has_bits);
          num_values_per_keypoint_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional int32 box_coord_offset = 12 [default = 0];
      case 12:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 96)) {
          _Internal::set_has_box_coord_offset(&has_bits);
          box_coord_offset_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional bool apply_exponential_on_box_size = 13 [default = false];
      case 13:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 104)) {
          _Internal::set_has_apply_exponential_on_box_size(&has_bits);
          apply_exponential_on_box_size_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional bool reverse_output_order = 14 [default = false];
      case 14:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 112)) {
          _Internal::set_has_reverse_output_order(&has_bits);
          reverse_output_order_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional bool sigmoid_score = 15 [default = false];
      case 15:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 120)) {
          _Internal::set_has_sigmoid_score(&has_bits);
          sigmoid_score_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional float score_clipping_thresh = 16;
      case 16:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 133)) {
          _Internal::set_has_score_clipping_thresh(&has_bits);
          score_clipping_thresh_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else goto handle_unusual;
        continue;
      // optional bool flip_vertically = 18 [default = false];
      case 18:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 144)) {
          _Internal::set_has_flip_vertically(&has_bits);
          flip_vertically_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional float min_score_thresh = 19;
      case 19:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 157)) {
          _Internal::set_has_min_score_thresh(&has_bits);
          min_score_thresh_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
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

::PROTOBUF_NAMESPACE_ID::uint8* TfLiteTensorsToDetectionsCalculatorOptions::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.TfLiteTensorsToDetectionsCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional int32 num_classes = 1;
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->_internal_num_classes(), target);
  }

  // optional int32 num_boxes = 2;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(2, this->_internal_num_boxes(), target);
  }

  // optional int32 num_coords = 3;
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(3, this->_internal_num_coords(), target);
  }

  // optional float x_scale = 4 [default = 0];
  if (cached_has_bits & 0x00000008u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(4, this->_internal_x_scale(), target);
  }

  // optional float y_scale = 5 [default = 0];
  if (cached_has_bits & 0x00000010u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(5, this->_internal_y_scale(), target);
  }

  // optional float w_scale = 6 [default = 0];
  if (cached_has_bits & 0x00000020u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(6, this->_internal_w_scale(), target);
  }

  // optional float h_scale = 7 [default = 0];
  if (cached_has_bits & 0x00000040u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(7, this->_internal_h_scale(), target);
  }

  // repeated int32 ignore_classes = 8;
  for (int i = 0, n = this->_internal_ignore_classes_size(); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(8, this->_internal_ignore_classes(i), target);
  }

  // optional int32 keypoint_coord_offset = 9;
  if (cached_has_bits & 0x00000080u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(9, this->_internal_keypoint_coord_offset(), target);
  }

  // optional int32 num_keypoints = 10 [default = 0];
  if (cached_has_bits & 0x00000100u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(10, this->_internal_num_keypoints(), target);
  }

  // optional int32 num_values_per_keypoint = 11 [default = 2];
  if (cached_has_bits & 0x00010000u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(11, this->_internal_num_values_per_keypoint(), target);
  }

  // optional int32 box_coord_offset = 12 [default = 0];
  if (cached_has_bits & 0x00000200u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(12, this->_internal_box_coord_offset(), target);
  }

  // optional bool apply_exponential_on_box_size = 13 [default = false];
  if (cached_has_bits & 0x00000400u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(13, this->_internal_apply_exponential_on_box_size(), target);
  }

  // optional bool reverse_output_order = 14 [default = false];
  if (cached_has_bits & 0x00000800u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(14, this->_internal_reverse_output_order(), target);
  }

  // optional bool sigmoid_score = 15 [default = false];
  if (cached_has_bits & 0x00001000u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(15, this->_internal_sigmoid_score(), target);
  }

  // optional float score_clipping_thresh = 16;
  if (cached_has_bits & 0x00004000u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(16, this->_internal_score_clipping_thresh(), target);
  }

  // optional bool flip_vertically = 18 [default = false];
  if (cached_has_bits & 0x00002000u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(18, this->_internal_flip_vertically(), target);
  }

  // optional float min_score_thresh = 19;
  if (cached_has_bits & 0x00008000u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(19, this->_internal_min_score_thresh(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.TfLiteTensorsToDetectionsCalculatorOptions)
  return target;
}

size_t TfLiteTensorsToDetectionsCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.TfLiteTensorsToDetectionsCalculatorOptions)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated int32 ignore_classes = 8;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      Int32Size(this->ignore_classes_);
    total_size += 1 *
                  ::PROTOBUF_NAMESPACE_ID::internal::FromIntSize(this->_internal_ignore_classes_size());
    total_size += data_size;
  }

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    // optional int32 num_classes = 1;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_num_classes());
    }

    // optional int32 num_boxes = 2;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_num_boxes());
    }

    // optional int32 num_coords = 3;
    if (cached_has_bits & 0x00000004u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_num_coords());
    }

    // optional float x_scale = 4 [default = 0];
    if (cached_has_bits & 0x00000008u) {
      total_size += 1 + 4;
    }

    // optional float y_scale = 5 [default = 0];
    if (cached_has_bits & 0x00000010u) {
      total_size += 1 + 4;
    }

    // optional float w_scale = 6 [default = 0];
    if (cached_has_bits & 0x00000020u) {
      total_size += 1 + 4;
    }

    // optional float h_scale = 7 [default = 0];
    if (cached_has_bits & 0x00000040u) {
      total_size += 1 + 4;
    }

    // optional int32 keypoint_coord_offset = 9;
    if (cached_has_bits & 0x00000080u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_keypoint_coord_offset());
    }

  }
  if (cached_has_bits & 0x0000ff00u) {
    // optional int32 num_keypoints = 10 [default = 0];
    if (cached_has_bits & 0x00000100u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_num_keypoints());
    }

    // optional int32 box_coord_offset = 12 [default = 0];
    if (cached_has_bits & 0x00000200u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->_internal_box_coord_offset());
    }

    // optional bool apply_exponential_on_box_size = 13 [default = false];
    if (cached_has_bits & 0x00000400u) {
      total_size += 1 + 1;
    }

    // optional bool reverse_output_order = 14 [default = false];
    if (cached_has_bits & 0x00000800u) {
      total_size += 1 + 1;
    }

    // optional bool sigmoid_score = 15 [default = false];
    if (cached_has_bits & 0x00001000u) {
      total_size += 1 + 1;
    }

    // optional bool flip_vertically = 18 [default = false];
    if (cached_has_bits & 0x00002000u) {
      total_size += 2 + 1;
    }

    // optional float score_clipping_thresh = 16;
    if (cached_has_bits & 0x00004000u) {
      total_size += 2 + 4;
    }

    // optional float min_score_thresh = 19;
    if (cached_has_bits & 0x00008000u) {
      total_size += 2 + 4;
    }

  }
  // optional int32 num_values_per_keypoint = 11 [default = 2];
  if (cached_has_bits & 0x00010000u) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
        this->_internal_num_values_per_keypoint());
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void TfLiteTensorsToDetectionsCalculatorOptions::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.TfLiteTensorsToDetectionsCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  const TfLiteTensorsToDetectionsCalculatorOptions* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<TfLiteTensorsToDetectionsCalculatorOptions>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.TfLiteTensorsToDetectionsCalculatorOptions)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.TfLiteTensorsToDetectionsCalculatorOptions)
    MergeFrom(*source);
  }
}

void TfLiteTensorsToDetectionsCalculatorOptions::MergeFrom(const TfLiteTensorsToDetectionsCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.TfLiteTensorsToDetectionsCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  ignore_classes_.MergeFrom(from.ignore_classes_);
  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x000000ffu) {
    if (cached_has_bits & 0x00000001u) {
      num_classes_ = from.num_classes_;
    }
    if (cached_has_bits & 0x00000002u) {
      num_boxes_ = from.num_boxes_;
    }
    if (cached_has_bits & 0x00000004u) {
      num_coords_ = from.num_coords_;
    }
    if (cached_has_bits & 0x00000008u) {
      x_scale_ = from.x_scale_;
    }
    if (cached_has_bits & 0x00000010u) {
      y_scale_ = from.y_scale_;
    }
    if (cached_has_bits & 0x00000020u) {
      w_scale_ = from.w_scale_;
    }
    if (cached_has_bits & 0x00000040u) {
      h_scale_ = from.h_scale_;
    }
    if (cached_has_bits & 0x00000080u) {
      keypoint_coord_offset_ = from.keypoint_coord_offset_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
  if (cached_has_bits & 0x0000ff00u) {
    if (cached_has_bits & 0x00000100u) {
      num_keypoints_ = from.num_keypoints_;
    }
    if (cached_has_bits & 0x00000200u) {
      box_coord_offset_ = from.box_coord_offset_;
    }
    if (cached_has_bits & 0x00000400u) {
      apply_exponential_on_box_size_ = from.apply_exponential_on_box_size_;
    }
    if (cached_has_bits & 0x00000800u) {
      reverse_output_order_ = from.reverse_output_order_;
    }
    if (cached_has_bits & 0x00001000u) {
      sigmoid_score_ = from.sigmoid_score_;
    }
    if (cached_has_bits & 0x00002000u) {
      flip_vertically_ = from.flip_vertically_;
    }
    if (cached_has_bits & 0x00004000u) {
      score_clipping_thresh_ = from.score_clipping_thresh_;
    }
    if (cached_has_bits & 0x00008000u) {
      min_score_thresh_ = from.min_score_thresh_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
  if (cached_has_bits & 0x00010000u) {
    _internal_set_num_values_per_keypoint(from._internal_num_values_per_keypoint());
  }
}

void TfLiteTensorsToDetectionsCalculatorOptions::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.TfLiteTensorsToDetectionsCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void TfLiteTensorsToDetectionsCalculatorOptions::CopyFrom(const TfLiteTensorsToDetectionsCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.TfLiteTensorsToDetectionsCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool TfLiteTensorsToDetectionsCalculatorOptions::IsInitialized() const {
  return true;
}

void TfLiteTensorsToDetectionsCalculatorOptions::InternalSwap(TfLiteTensorsToDetectionsCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  ignore_classes_.InternalSwap(&other->ignore_classes_);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(TfLiteTensorsToDetectionsCalculatorOptions, min_score_thresh_)
      + sizeof(TfLiteTensorsToDetectionsCalculatorOptions::min_score_thresh_)
      - PROTOBUF_FIELD_OFFSET(TfLiteTensorsToDetectionsCalculatorOptions, num_classes_)>(
          reinterpret_cast<char*>(&num_classes_),
          reinterpret_cast<char*>(&other->num_classes_));
  swap(num_values_per_keypoint_, other->num_values_per_keypoint_);
}

::PROTOBUF_NAMESPACE_ID::Metadata TfLiteTensorsToDetectionsCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2ftflite_2ftflite_5ftensors_5fto_5fdetections_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int TfLiteTensorsToDetectionsCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions >, 11, false >
  TfLiteTensorsToDetectionsCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
