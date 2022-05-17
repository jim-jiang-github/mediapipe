// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/util/detection_label_id_to_text_calculator.proto

#include "mediapipe/calculators/util/detection_label_id_to_text_calculator.pb.h"

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
constexpr DetectionLabelIdToTextCalculatorOptions::DetectionLabelIdToTextCalculatorOptions(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : label_()
  , label_map_path_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , label_map_(nullptr)
  , keep_label_id_(false){}
struct DetectionLabelIdToTextCalculatorOptionsDefaultTypeInternal {
  constexpr DetectionLabelIdToTextCalculatorOptionsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~DetectionLabelIdToTextCalculatorOptionsDefaultTypeInternal() {}
  union {
    DetectionLabelIdToTextCalculatorOptions _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT DetectionLabelIdToTextCalculatorOptionsDefaultTypeInternal _DetectionLabelIdToTextCalculatorOptions_default_instance_;
}  // namespace mediapipe
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionLabelIdToTextCalculatorOptions, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionLabelIdToTextCalculatorOptions, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionLabelIdToTextCalculatorOptions, label_map_path_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionLabelIdToTextCalculatorOptions, label_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionLabelIdToTextCalculatorOptions, keep_label_id_),
  PROTOBUF_FIELD_OFFSET(::mediapipe::DetectionLabelIdToTextCalculatorOptions, label_map_),
  0,
  ~0u,
  2,
  1,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 9, sizeof(::mediapipe::DetectionLabelIdToTextCalculatorOptions)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::mediapipe::_DetectionLabelIdToTextCalculatorOptions_default_instance_),
};

const char descriptor_table_protodef_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\nFmediapipe/calculators/util/detection_l"
  "abel_id_to_text_calculator.proto\022\tmediap"
  "ipe\032$mediapipe/framework/calculator.prot"
  "o\032\036mediapipe/util/label_map.proto\"\361\001\n\'De"
  "tectionLabelIdToTextCalculatorOptions\022\026\n"
  "\016label_map_path\030\001 \001(\t\022\r\n\005label\030\002 \003(\t\022\025\n\r"
  "keep_label_id\030\003 \001(\010\022&\n\tlabel_map\030\004 \001(\0132\023"
  ".mediapipe.LabelMap2`\n\003ext\022\034.mediapipe.C"
  "alculatorOptions\030\260\213\216x \001(\01322.mediapipe.De"
  "tectionLabelIdToTextCalculatorOptions"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto_deps[2] = {
  &::descriptor_table_mediapipe_2fframework_2fcalculator_2eproto,
  &::descriptor_table_mediapipe_2futil_2flabel_5fmap_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto = {
  false, false, 397, descriptor_table_protodef_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto, "mediapipe/calculators/util/detection_label_id_to_text_calculator.proto", 
  &descriptor_table_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto_once, descriptor_table_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto_deps, 2, 1,
  schemas, file_default_instances, TableStruct_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto::offsets,
  file_level_metadata_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto, file_level_enum_descriptors_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto, file_level_service_descriptors_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto_getter() {
  return &descriptor_table_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto(&descriptor_table_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto);
namespace mediapipe {

// ===================================================================

class DetectionLabelIdToTextCalculatorOptions::_Internal {
 public:
  using HasBits = decltype(std::declval<DetectionLabelIdToTextCalculatorOptions>()._has_bits_);
  static void set_has_label_map_path(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static void set_has_keep_label_id(HasBits* has_bits) {
    (*has_bits)[0] |= 4u;
  }
  static const ::mediapipe::LabelMap& label_map(const DetectionLabelIdToTextCalculatorOptions* msg);
  static void set_has_label_map(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
};

const ::mediapipe::LabelMap&
DetectionLabelIdToTextCalculatorOptions::_Internal::label_map(const DetectionLabelIdToTextCalculatorOptions* msg) {
  return *msg->label_map_;
}
void DetectionLabelIdToTextCalculatorOptions::clear_label_map() {
  if (label_map_ != nullptr) label_map_->Clear();
  _has_bits_[0] &= ~0x00000002u;
}
DetectionLabelIdToTextCalculatorOptions::DetectionLabelIdToTextCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena),
  label_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:mediapipe.DetectionLabelIdToTextCalculatorOptions)
}
DetectionLabelIdToTextCalculatorOptions::DetectionLabelIdToTextCalculatorOptions(const DetectionLabelIdToTextCalculatorOptions& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_),
      label_(from.label_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  label_map_path_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from._internal_has_label_map_path()) {
    label_map_path_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_label_map_path(), 
      GetArena());
  }
  if (from._internal_has_label_map()) {
    label_map_ = new ::mediapipe::LabelMap(*from.label_map_);
  } else {
    label_map_ = nullptr;
  }
  keep_label_id_ = from.keep_label_id_;
  // @@protoc_insertion_point(copy_constructor:mediapipe.DetectionLabelIdToTextCalculatorOptions)
}

void DetectionLabelIdToTextCalculatorOptions::SharedCtor() {
label_map_path_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
::memset(reinterpret_cast<char*>(this) + static_cast<size_t>(
    reinterpret_cast<char*>(&label_map_) - reinterpret_cast<char*>(this)),
    0, static_cast<size_t>(reinterpret_cast<char*>(&keep_label_id_) -
    reinterpret_cast<char*>(&label_map_)) + sizeof(keep_label_id_));
}

DetectionLabelIdToTextCalculatorOptions::~DetectionLabelIdToTextCalculatorOptions() {
  // @@protoc_insertion_point(destructor:mediapipe.DetectionLabelIdToTextCalculatorOptions)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void DetectionLabelIdToTextCalculatorOptions::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
  label_map_path_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (this != internal_default_instance()) delete label_map_;
}

void DetectionLabelIdToTextCalculatorOptions::ArenaDtor(void* object) {
  DetectionLabelIdToTextCalculatorOptions* _this = reinterpret_cast< DetectionLabelIdToTextCalculatorOptions* >(object);
  (void)_this;
}
void DetectionLabelIdToTextCalculatorOptions::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void DetectionLabelIdToTextCalculatorOptions::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void DetectionLabelIdToTextCalculatorOptions::Clear() {
// @@protoc_insertion_point(message_clear_start:mediapipe.DetectionLabelIdToTextCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  label_.Clear();
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    if (cached_has_bits & 0x00000001u) {
      label_map_path_.ClearNonDefaultToEmpty();
    }
    if (cached_has_bits & 0x00000002u) {
      GOOGLE_DCHECK(label_map_ != nullptr);
      label_map_->Clear();
    }
  }
  keep_label_id_ = false;
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* DetectionLabelIdToTextCalculatorOptions::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional string label_map_path = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          auto str = _internal_mutable_label_map_path();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          #ifndef NDEBUG
          ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "mediapipe.DetectionLabelIdToTextCalculatorOptions.label_map_path");
          #endif  // !NDEBUG
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated string label = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          ptr -= 1;
          do {
            ptr += 1;
            auto str = _internal_add_label();
            ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
            #ifndef NDEBUG
            ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "mediapipe.DetectionLabelIdToTextCalculatorOptions.label");
            #endif  // !NDEBUG
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<18>(ptr));
        } else goto handle_unusual;
        continue;
      // optional bool keep_label_id = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          _Internal::set_has_keep_label_id(&has_bits);
          keep_label_id_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional .mediapipe.LabelMap label_map = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 34)) {
          ptr = ctx->ParseMessage(_internal_mutable_label_map(), ptr);
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

::PROTOBUF_NAMESPACE_ID::uint8* DetectionLabelIdToTextCalculatorOptions::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:mediapipe.DetectionLabelIdToTextCalculatorOptions)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional string label_map_path = 1;
  if (cached_has_bits & 0x00000001u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->_internal_label_map_path().data(), static_cast<int>(this->_internal_label_map_path().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "mediapipe.DetectionLabelIdToTextCalculatorOptions.label_map_path");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_label_map_path(), target);
  }

  // repeated string label = 2;
  for (int i = 0, n = this->_internal_label_size(); i < n; i++) {
    const auto& s = this->_internal_label(i);
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      s.data(), static_cast<int>(s.length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "mediapipe.DetectionLabelIdToTextCalculatorOptions.label");
    target = stream->WriteString(2, s, target);
  }

  // optional bool keep_label_id = 3;
  if (cached_has_bits & 0x00000004u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(3, this->_internal_keep_label_id(), target);
  }

  // optional .mediapipe.LabelMap label_map = 4;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(
        4, _Internal::label_map(this), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:mediapipe.DetectionLabelIdToTextCalculatorOptions)
  return target;
}

size_t DetectionLabelIdToTextCalculatorOptions::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:mediapipe.DetectionLabelIdToTextCalculatorOptions)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated string label = 2;
  total_size += 1 *
      ::PROTOBUF_NAMESPACE_ID::internal::FromIntSize(label_.size());
  for (int i = 0, n = label_.size(); i < n; i++) {
    total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
      label_.Get(i));
  }

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    // optional string label_map_path = 1;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_label_map_path());
    }

    // optional .mediapipe.LabelMap label_map = 4;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *label_map_);
    }

    // optional bool keep_label_id = 3;
    if (cached_has_bits & 0x00000004u) {
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

void DetectionLabelIdToTextCalculatorOptions::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:mediapipe.DetectionLabelIdToTextCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  const DetectionLabelIdToTextCalculatorOptions* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<DetectionLabelIdToTextCalculatorOptions>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:mediapipe.DetectionLabelIdToTextCalculatorOptions)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:mediapipe.DetectionLabelIdToTextCalculatorOptions)
    MergeFrom(*source);
  }
}

void DetectionLabelIdToTextCalculatorOptions::MergeFrom(const DetectionLabelIdToTextCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:mediapipe.DetectionLabelIdToTextCalculatorOptions)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  label_.MergeFrom(from.label_);
  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x00000007u) {
    if (cached_has_bits & 0x00000001u) {
      _internal_set_label_map_path(from._internal_label_map_path());
    }
    if (cached_has_bits & 0x00000002u) {
      _internal_mutable_label_map()->::mediapipe::LabelMap::MergeFrom(from._internal_label_map());
    }
    if (cached_has_bits & 0x00000004u) {
      keep_label_id_ = from.keep_label_id_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void DetectionLabelIdToTextCalculatorOptions::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:mediapipe.DetectionLabelIdToTextCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void DetectionLabelIdToTextCalculatorOptions::CopyFrom(const DetectionLabelIdToTextCalculatorOptions& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:mediapipe.DetectionLabelIdToTextCalculatorOptions)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool DetectionLabelIdToTextCalculatorOptions::IsInitialized() const {
  return true;
}

void DetectionLabelIdToTextCalculatorOptions::InternalSwap(DetectionLabelIdToTextCalculatorOptions* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  label_.InternalSwap(&other->label_);
  label_map_path_.Swap(&other->label_map_path_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(DetectionLabelIdToTextCalculatorOptions, keep_label_id_)
      + sizeof(DetectionLabelIdToTextCalculatorOptions::keep_label_id_)
      - PROTOBUF_FIELD_OFFSET(DetectionLabelIdToTextCalculatorOptions, label_map_)>(
          reinterpret_cast<char*>(&label_map_),
          reinterpret_cast<char*>(&other->label_map_));
}

::PROTOBUF_NAMESPACE_ID::Metadata DetectionLabelIdToTextCalculatorOptions::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto_getter, &descriptor_table_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto_once,
      file_level_metadata_mediapipe_2fcalculators_2futil_2fdetection_5flabel_5fid_5fto_5ftext_5fcalculator_2eproto[0]);
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int DetectionLabelIdToTextCalculatorOptions::kExtFieldNumber;
#endif
PROTOBUF_ATTRIBUTE_INIT_PRIORITY ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
    ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::DetectionLabelIdToTextCalculatorOptions >, 11, false >
  DetectionLabelIdToTextCalculatorOptions::ext(kExtFieldNumber, ::mediapipe::DetectionLabelIdToTextCalculatorOptions::default_instance());

// @@protoc_insertion_point(namespace_scope)
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::mediapipe::DetectionLabelIdToTextCalculatorOptions* Arena::CreateMaybeMessage< ::mediapipe::DetectionLabelIdToTextCalculatorOptions >(Arena* arena) {
  return Arena::CreateMessageInternal< ::mediapipe::DetectionLabelIdToTextCalculatorOptions >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
