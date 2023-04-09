// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/util/labels_to_render_data_calculator.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2futil_2flabels_5fto_5frender_5fdata_5fcalculator_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2futil_2flabels_5fto_5frender_5fdata_5fcalculator_2eproto

#include <limits>
#include <string>

#include <x/google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3015000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3015008 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <x/google/protobuf/port_undef.inc>
#include <x/google/protobuf/io/coded_stream.h>
#include <x/google/protobuf/arena.h>
#include <x/google/protobuf/arenastring.h>
#include <x/google/protobuf/generated_message_table_driven.h>
#include <x/google/protobuf/generated_message_util.h>
#include <x/google/protobuf/metadata_lite.h>
#include <x/google/protobuf/generated_message_reflection.h>
#include <x/google/protobuf/message.h>
#include <x/google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <x/google/protobuf/extension_set.h>  // IWYU pragma: export
#include <x/google/protobuf/generated_enum_reflection.h>
#include <x/google/protobuf/unknown_field_set.h>
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/util/color.pb.h"
// @@protoc_insertion_point(includes)
#include <x/google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fcalculators_2futil_2flabels_5fto_5frender_5fdata_5fcalculator_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fcalculators_2futil_2flabels_5fto_5frender_5fdata_5fcalculator_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[1]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2futil_2flabels_5fto_5frender_5fdata_5fcalculator_2eproto;
namespace mediapipe {
class LabelsToRenderDataCalculatorOptions;
struct LabelsToRenderDataCalculatorOptionsDefaultTypeInternal;
extern LabelsToRenderDataCalculatorOptionsDefaultTypeInternal _LabelsToRenderDataCalculatorOptions_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::LabelsToRenderDataCalculatorOptions* Arena::CreateMaybeMessage<::mediapipe::LabelsToRenderDataCalculatorOptions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

enum LabelsToRenderDataCalculatorOptions_Location : int {
  LabelsToRenderDataCalculatorOptions_Location_TOP_LEFT = 0,
  LabelsToRenderDataCalculatorOptions_Location_BOTTOM_LEFT = 1
};
bool LabelsToRenderDataCalculatorOptions_Location_IsValid(int value);
constexpr LabelsToRenderDataCalculatorOptions_Location LabelsToRenderDataCalculatorOptions_Location_Location_MIN = LabelsToRenderDataCalculatorOptions_Location_TOP_LEFT;
constexpr LabelsToRenderDataCalculatorOptions_Location LabelsToRenderDataCalculatorOptions_Location_Location_MAX = LabelsToRenderDataCalculatorOptions_Location_BOTTOM_LEFT;
constexpr int LabelsToRenderDataCalculatorOptions_Location_Location_ARRAYSIZE = LabelsToRenderDataCalculatorOptions_Location_Location_MAX + 1;

const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* LabelsToRenderDataCalculatorOptions_Location_descriptor();
template<typename T>
inline const std::string& LabelsToRenderDataCalculatorOptions_Location_Name(T enum_t_value) {
  static_assert(::std::is_same<T, LabelsToRenderDataCalculatorOptions_Location>::value ||
    ::std::is_integral<T>::value,
    "Incorrect type passed to function LabelsToRenderDataCalculatorOptions_Location_Name.");
  return ::PROTOBUF_NAMESPACE_ID::internal::NameOfEnum(
    LabelsToRenderDataCalculatorOptions_Location_descriptor(), enum_t_value);
}
inline bool LabelsToRenderDataCalculatorOptions_Location_Parse(
    ::PROTOBUF_NAMESPACE_ID::ConstStringParam name, LabelsToRenderDataCalculatorOptions_Location* value) {
  return ::PROTOBUF_NAMESPACE_ID::internal::ParseNamedEnum<LabelsToRenderDataCalculatorOptions_Location>(
    LabelsToRenderDataCalculatorOptions_Location_descriptor(), name, value);
}
// ===================================================================

class LabelsToRenderDataCalculatorOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.LabelsToRenderDataCalculatorOptions) */ {
 public:
  inline LabelsToRenderDataCalculatorOptions() : LabelsToRenderDataCalculatorOptions(nullptr) {}
  ~LabelsToRenderDataCalculatorOptions() override;
  explicit constexpr LabelsToRenderDataCalculatorOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  LabelsToRenderDataCalculatorOptions(const LabelsToRenderDataCalculatorOptions& from);
  LabelsToRenderDataCalculatorOptions(LabelsToRenderDataCalculatorOptions&& from) noexcept
    : LabelsToRenderDataCalculatorOptions() {
    *this = ::std::move(from);
  }

  inline LabelsToRenderDataCalculatorOptions& operator=(const LabelsToRenderDataCalculatorOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline LabelsToRenderDataCalculatorOptions& operator=(LabelsToRenderDataCalculatorOptions&& from) noexcept {
    if (GetArena() == from.GetArena()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance);
  }
  inline ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const LabelsToRenderDataCalculatorOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const LabelsToRenderDataCalculatorOptions* internal_default_instance() {
    return reinterpret_cast<const LabelsToRenderDataCalculatorOptions*>(
               &_LabelsToRenderDataCalculatorOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(LabelsToRenderDataCalculatorOptions& a, LabelsToRenderDataCalculatorOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(LabelsToRenderDataCalculatorOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(LabelsToRenderDataCalculatorOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline LabelsToRenderDataCalculatorOptions* New() const final {
    return CreateMaybeMessage<LabelsToRenderDataCalculatorOptions>(nullptr);
  }

  LabelsToRenderDataCalculatorOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<LabelsToRenderDataCalculatorOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const LabelsToRenderDataCalculatorOptions& from);
  void MergeFrom(const LabelsToRenderDataCalculatorOptions& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(LabelsToRenderDataCalculatorOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.LabelsToRenderDataCalculatorOptions";
  }
  protected:
  explicit LabelsToRenderDataCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef LabelsToRenderDataCalculatorOptions_Location Location;
  static constexpr Location TOP_LEFT =
    LabelsToRenderDataCalculatorOptions_Location_TOP_LEFT;
  static constexpr Location BOTTOM_LEFT =
    LabelsToRenderDataCalculatorOptions_Location_BOTTOM_LEFT;
  static inline bool Location_IsValid(int value) {
    return LabelsToRenderDataCalculatorOptions_Location_IsValid(value);
  }
  static constexpr Location Location_MIN =
    LabelsToRenderDataCalculatorOptions_Location_Location_MIN;
  static constexpr Location Location_MAX =
    LabelsToRenderDataCalculatorOptions_Location_Location_MAX;
  static constexpr int Location_ARRAYSIZE =
    LabelsToRenderDataCalculatorOptions_Location_Location_ARRAYSIZE;
  static inline const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor*
  Location_descriptor() {
    return LabelsToRenderDataCalculatorOptions_Location_descriptor();
  }
  template<typename T>
  static inline const std::string& Location_Name(T enum_t_value) {
    static_assert(::std::is_same<T, Location>::value ||
      ::std::is_integral<T>::value,
      "Incorrect type passed to function Location_Name.");
    return LabelsToRenderDataCalculatorOptions_Location_Name(enum_t_value);
  }
  static inline bool Location_Parse(::PROTOBUF_NAMESPACE_ID::ConstStringParam name,
      Location* value) {
    return LabelsToRenderDataCalculatorOptions_Location_Parse(name, value);
  }

  // accessors -------------------------------------------------------

  enum : int {
    kColorFieldNumber = 1,
    kOutlineColorFieldNumber = 12,
    kFontFaceFieldNumber = 5,
    kLocationFieldNumber = 6,
    kHorizontalOffsetPxFieldNumber = 7,
    kVerticalOffsetPxFieldNumber = 8,
    kOutlineThicknessFieldNumber = 11,
    kUseDisplayNameFieldNumber = 9,
    kDisplayClassificationScoreFieldNumber = 10,
    kThicknessFieldNumber = 2,
    kFontHeightPxFieldNumber = 3,
    kMaxNumLabelsFieldNumber = 4,
  };
  // repeated .mediapipe.Color color = 1;
  int color_size() const;
  private:
  int _internal_color_size() const;
  public:
  void clear_color();
  ::mediapipe::Color* mutable_color(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::Color >*
      mutable_color();
  private:
  const ::mediapipe::Color& _internal_color(int index) const;
  ::mediapipe::Color* _internal_add_color();
  public:
  const ::mediapipe::Color& color(int index) const;
  ::mediapipe::Color* add_color();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::Color >&
      color() const;

  // repeated .mediapipe.Color outline_color = 12;
  int outline_color_size() const;
  private:
  int _internal_outline_color_size() const;
  public:
  void clear_outline_color();
  ::mediapipe::Color* mutable_outline_color(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::Color >*
      mutable_outline_color();
  private:
  const ::mediapipe::Color& _internal_outline_color(int index) const;
  ::mediapipe::Color* _internal_add_outline_color();
  public:
  const ::mediapipe::Color& outline_color(int index) const;
  ::mediapipe::Color* add_outline_color();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::Color >&
      outline_color() const;

  // optional int32 font_face = 5 [default = 0];
  bool has_font_face() const;
  private:
  bool _internal_has_font_face() const;
  public:
  void clear_font_face();
  ::PROTOBUF_NAMESPACE_ID::int32 font_face() const;
  void set_font_face(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_font_face() const;
  void _internal_set_font_face(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional .mediapipe.LabelsToRenderDataCalculatorOptions.Location location = 6 [default = TOP_LEFT];
  bool has_location() const;
  private:
  bool _internal_has_location() const;
  public:
  void clear_location();
  ::mediapipe::LabelsToRenderDataCalculatorOptions_Location location() const;
  void set_location(::mediapipe::LabelsToRenderDataCalculatorOptions_Location value);
  private:
  ::mediapipe::LabelsToRenderDataCalculatorOptions_Location _internal_location() const;
  void _internal_set_location(::mediapipe::LabelsToRenderDataCalculatorOptions_Location value);
  public:

  // optional int32 horizontal_offset_px = 7 [default = 0];
  bool has_horizontal_offset_px() const;
  private:
  bool _internal_has_horizontal_offset_px() const;
  public:
  void clear_horizontal_offset_px();
  ::PROTOBUF_NAMESPACE_ID::int32 horizontal_offset_px() const;
  void set_horizontal_offset_px(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_horizontal_offset_px() const;
  void _internal_set_horizontal_offset_px(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional int32 vertical_offset_px = 8 [default = 0];
  bool has_vertical_offset_px() const;
  private:
  bool _internal_has_vertical_offset_px() const;
  public:
  void clear_vertical_offset_px();
  ::PROTOBUF_NAMESPACE_ID::int32 vertical_offset_px() const;
  void set_vertical_offset_px(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_vertical_offset_px() const;
  void _internal_set_vertical_offset_px(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional double outline_thickness = 11;
  bool has_outline_thickness() const;
  private:
  bool _internal_has_outline_thickness() const;
  public:
  void clear_outline_thickness();
  double outline_thickness() const;
  void set_outline_thickness(double value);
  private:
  double _internal_outline_thickness() const;
  void _internal_set_outline_thickness(double value);
  public:

  // optional bool use_display_name = 9 [default = false];
  bool has_use_display_name() const;
  private:
  bool _internal_has_use_display_name() const;
  public:
  void clear_use_display_name();
  bool use_display_name() const;
  void set_use_display_name(bool value);
  private:
  bool _internal_use_display_name() const;
  void _internal_set_use_display_name(bool value);
  public:

  // optional bool display_classification_score = 10 [default = false];
  bool has_display_classification_score() const;
  private:
  bool _internal_has_display_classification_score() const;
  public:
  void clear_display_classification_score();
  bool display_classification_score() const;
  void set_display_classification_score(bool value);
  private:
  bool _internal_display_classification_score() const;
  void _internal_set_display_classification_score(bool value);
  public:

  // optional double thickness = 2 [default = 2];
  bool has_thickness() const;
  private:
  bool _internal_has_thickness() const;
  public:
  void clear_thickness();
  double thickness() const;
  void set_thickness(double value);
  private:
  double _internal_thickness() const;
  void _internal_set_thickness(double value);
  public:

  // optional int32 font_height_px = 3 [default = 50];
  bool has_font_height_px() const;
  private:
  bool _internal_has_font_height_px() const;
  public:
  void clear_font_height_px();
  ::PROTOBUF_NAMESPACE_ID::int32 font_height_px() const;
  void set_font_height_px(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_font_height_px() const;
  void _internal_set_font_height_px(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional int32 max_num_labels = 4 [default = 1];
  bool has_max_num_labels() const;
  private:
  bool _internal_has_max_num_labels() const;
  public:
  void clear_max_num_labels();
  ::PROTOBUF_NAMESPACE_ID::int32 max_num_labels() const;
  void set_max_num_labels(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_max_num_labels() const;
  void _internal_set_max_num_labels(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  static const int kExtFieldNumber = 271660364;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::LabelsToRenderDataCalculatorOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.LabelsToRenderDataCalculatorOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::Color > color_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::Color > outline_color_;
  ::PROTOBUF_NAMESPACE_ID::int32 font_face_;
  int location_;
  ::PROTOBUF_NAMESPACE_ID::int32 horizontal_offset_px_;
  ::PROTOBUF_NAMESPACE_ID::int32 vertical_offset_px_;
  double outline_thickness_;
  bool use_display_name_;
  bool display_classification_score_;
  double thickness_;
  ::PROTOBUF_NAMESPACE_ID::int32 font_height_px_;
  ::PROTOBUF_NAMESPACE_ID::int32 max_num_labels_;
  friend struct ::TableStruct_mediapipe_2fcalculators_2futil_2flabels_5fto_5frender_5fdata_5fcalculator_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// LabelsToRenderDataCalculatorOptions

// repeated .mediapipe.Color color = 1;
inline int LabelsToRenderDataCalculatorOptions::_internal_color_size() const {
  return color_.size();
}
inline int LabelsToRenderDataCalculatorOptions::color_size() const {
  return _internal_color_size();
}
inline ::mediapipe::Color* LabelsToRenderDataCalculatorOptions::mutable_color(int index) {
  // @@protoc_insertion_point(field_mutable:mediapipe.LabelsToRenderDataCalculatorOptions.color)
  return color_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::Color >*
LabelsToRenderDataCalculatorOptions::mutable_color() {
  // @@protoc_insertion_point(field_mutable_list:mediapipe.LabelsToRenderDataCalculatorOptions.color)
  return &color_;
}
inline const ::mediapipe::Color& LabelsToRenderDataCalculatorOptions::_internal_color(int index) const {
  return color_.Get(index);
}
inline const ::mediapipe::Color& LabelsToRenderDataCalculatorOptions::color(int index) const {
  // @@protoc_insertion_point(field_get:mediapipe.LabelsToRenderDataCalculatorOptions.color)
  return _internal_color(index);
}
inline ::mediapipe::Color* LabelsToRenderDataCalculatorOptions::_internal_add_color() {
  return color_.Add();
}
inline ::mediapipe::Color* LabelsToRenderDataCalculatorOptions::add_color() {
  // @@protoc_insertion_point(field_add:mediapipe.LabelsToRenderDataCalculatorOptions.color)
  return _internal_add_color();
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::Color >&
LabelsToRenderDataCalculatorOptions::color() const {
  // @@protoc_insertion_point(field_list:mediapipe.LabelsToRenderDataCalculatorOptions.color)
  return color_;
}

// optional double thickness = 2 [default = 2];
inline bool LabelsToRenderDataCalculatorOptions::_internal_has_thickness() const {
  bool value = (_has_bits_[0] & 0x00000080u) != 0;
  return value;
}
inline bool LabelsToRenderDataCalculatorOptions::has_thickness() const {
  return _internal_has_thickness();
}
inline void LabelsToRenderDataCalculatorOptions::clear_thickness() {
  thickness_ = 2;
  _has_bits_[0] &= ~0x00000080u;
}
inline double LabelsToRenderDataCalculatorOptions::_internal_thickness() const {
  return thickness_;
}
inline double LabelsToRenderDataCalculatorOptions::thickness() const {
  // @@protoc_insertion_point(field_get:mediapipe.LabelsToRenderDataCalculatorOptions.thickness)
  return _internal_thickness();
}
inline void LabelsToRenderDataCalculatorOptions::_internal_set_thickness(double value) {
  _has_bits_[0] |= 0x00000080u;
  thickness_ = value;
}
inline void LabelsToRenderDataCalculatorOptions::set_thickness(double value) {
  _internal_set_thickness(value);
  // @@protoc_insertion_point(field_set:mediapipe.LabelsToRenderDataCalculatorOptions.thickness)
}

// repeated .mediapipe.Color outline_color = 12;
inline int LabelsToRenderDataCalculatorOptions::_internal_outline_color_size() const {
  return outline_color_.size();
}
inline int LabelsToRenderDataCalculatorOptions::outline_color_size() const {
  return _internal_outline_color_size();
}
inline ::mediapipe::Color* LabelsToRenderDataCalculatorOptions::mutable_outline_color(int index) {
  // @@protoc_insertion_point(field_mutable:mediapipe.LabelsToRenderDataCalculatorOptions.outline_color)
  return outline_color_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::Color >*
LabelsToRenderDataCalculatorOptions::mutable_outline_color() {
  // @@protoc_insertion_point(field_mutable_list:mediapipe.LabelsToRenderDataCalculatorOptions.outline_color)
  return &outline_color_;
}
inline const ::mediapipe::Color& LabelsToRenderDataCalculatorOptions::_internal_outline_color(int index) const {
  return outline_color_.Get(index);
}
inline const ::mediapipe::Color& LabelsToRenderDataCalculatorOptions::outline_color(int index) const {
  // @@protoc_insertion_point(field_get:mediapipe.LabelsToRenderDataCalculatorOptions.outline_color)
  return _internal_outline_color(index);
}
inline ::mediapipe::Color* LabelsToRenderDataCalculatorOptions::_internal_add_outline_color() {
  return outline_color_.Add();
}
inline ::mediapipe::Color* LabelsToRenderDataCalculatorOptions::add_outline_color() {
  // @@protoc_insertion_point(field_add:mediapipe.LabelsToRenderDataCalculatorOptions.outline_color)
  return _internal_add_outline_color();
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::Color >&
LabelsToRenderDataCalculatorOptions::outline_color() const {
  // @@protoc_insertion_point(field_list:mediapipe.LabelsToRenderDataCalculatorOptions.outline_color)
  return outline_color_;
}

// optional double outline_thickness = 11;
inline bool LabelsToRenderDataCalculatorOptions::_internal_has_outline_thickness() const {
  bool value = (_has_bits_[0] & 0x00000010u) != 0;
  return value;
}
inline bool LabelsToRenderDataCalculatorOptions::has_outline_thickness() const {
  return _internal_has_outline_thickness();
}
inline void LabelsToRenderDataCalculatorOptions::clear_outline_thickness() {
  outline_thickness_ = 0;
  _has_bits_[0] &= ~0x00000010u;
}
inline double LabelsToRenderDataCalculatorOptions::_internal_outline_thickness() const {
  return outline_thickness_;
}
inline double LabelsToRenderDataCalculatorOptions::outline_thickness() const {
  // @@protoc_insertion_point(field_get:mediapipe.LabelsToRenderDataCalculatorOptions.outline_thickness)
  return _internal_outline_thickness();
}
inline void LabelsToRenderDataCalculatorOptions::_internal_set_outline_thickness(double value) {
  _has_bits_[0] |= 0x00000010u;
  outline_thickness_ = value;
}
inline void LabelsToRenderDataCalculatorOptions::set_outline_thickness(double value) {
  _internal_set_outline_thickness(value);
  // @@protoc_insertion_point(field_set:mediapipe.LabelsToRenderDataCalculatorOptions.outline_thickness)
}

// optional int32 font_height_px = 3 [default = 50];
inline bool LabelsToRenderDataCalculatorOptions::_internal_has_font_height_px() const {
  bool value = (_has_bits_[0] & 0x00000100u) != 0;
  return value;
}
inline bool LabelsToRenderDataCalculatorOptions::has_font_height_px() const {
  return _internal_has_font_height_px();
}
inline void LabelsToRenderDataCalculatorOptions::clear_font_height_px() {
  font_height_px_ = 50;
  _has_bits_[0] &= ~0x00000100u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 LabelsToRenderDataCalculatorOptions::_internal_font_height_px() const {
  return font_height_px_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 LabelsToRenderDataCalculatorOptions::font_height_px() const {
  // @@protoc_insertion_point(field_get:mediapipe.LabelsToRenderDataCalculatorOptions.font_height_px)
  return _internal_font_height_px();
}
inline void LabelsToRenderDataCalculatorOptions::_internal_set_font_height_px(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000100u;
  font_height_px_ = value;
}
inline void LabelsToRenderDataCalculatorOptions::set_font_height_px(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_font_height_px(value);
  // @@protoc_insertion_point(field_set:mediapipe.LabelsToRenderDataCalculatorOptions.font_height_px)
}

// optional int32 horizontal_offset_px = 7 [default = 0];
inline bool LabelsToRenderDataCalculatorOptions::_internal_has_horizontal_offset_px() const {
  bool value = (_has_bits_[0] & 0x00000004u) != 0;
  return value;
}
inline bool LabelsToRenderDataCalculatorOptions::has_horizontal_offset_px() const {
  return _internal_has_horizontal_offset_px();
}
inline void LabelsToRenderDataCalculatorOptions::clear_horizontal_offset_px() {
  horizontal_offset_px_ = 0;
  _has_bits_[0] &= ~0x00000004u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 LabelsToRenderDataCalculatorOptions::_internal_horizontal_offset_px() const {
  return horizontal_offset_px_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 LabelsToRenderDataCalculatorOptions::horizontal_offset_px() const {
  // @@protoc_insertion_point(field_get:mediapipe.LabelsToRenderDataCalculatorOptions.horizontal_offset_px)
  return _internal_horizontal_offset_px();
}
inline void LabelsToRenderDataCalculatorOptions::_internal_set_horizontal_offset_px(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000004u;
  horizontal_offset_px_ = value;
}
inline void LabelsToRenderDataCalculatorOptions::set_horizontal_offset_px(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_horizontal_offset_px(value);
  // @@protoc_insertion_point(field_set:mediapipe.LabelsToRenderDataCalculatorOptions.horizontal_offset_px)
}

// optional int32 vertical_offset_px = 8 [default = 0];
inline bool LabelsToRenderDataCalculatorOptions::_internal_has_vertical_offset_px() const {
  bool value = (_has_bits_[0] & 0x00000008u) != 0;
  return value;
}
inline bool LabelsToRenderDataCalculatorOptions::has_vertical_offset_px() const {
  return _internal_has_vertical_offset_px();
}
inline void LabelsToRenderDataCalculatorOptions::clear_vertical_offset_px() {
  vertical_offset_px_ = 0;
  _has_bits_[0] &= ~0x00000008u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 LabelsToRenderDataCalculatorOptions::_internal_vertical_offset_px() const {
  return vertical_offset_px_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 LabelsToRenderDataCalculatorOptions::vertical_offset_px() const {
  // @@protoc_insertion_point(field_get:mediapipe.LabelsToRenderDataCalculatorOptions.vertical_offset_px)
  return _internal_vertical_offset_px();
}
inline void LabelsToRenderDataCalculatorOptions::_internal_set_vertical_offset_px(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000008u;
  vertical_offset_px_ = value;
}
inline void LabelsToRenderDataCalculatorOptions::set_vertical_offset_px(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_vertical_offset_px(value);
  // @@protoc_insertion_point(field_set:mediapipe.LabelsToRenderDataCalculatorOptions.vertical_offset_px)
}

// optional int32 max_num_labels = 4 [default = 1];
inline bool LabelsToRenderDataCalculatorOptions::_internal_has_max_num_labels() const {
  bool value = (_has_bits_[0] & 0x00000200u) != 0;
  return value;
}
inline bool LabelsToRenderDataCalculatorOptions::has_max_num_labels() const {
  return _internal_has_max_num_labels();
}
inline void LabelsToRenderDataCalculatorOptions::clear_max_num_labels() {
  max_num_labels_ = 1;
  _has_bits_[0] &= ~0x00000200u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 LabelsToRenderDataCalculatorOptions::_internal_max_num_labels() const {
  return max_num_labels_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 LabelsToRenderDataCalculatorOptions::max_num_labels() const {
  // @@protoc_insertion_point(field_get:mediapipe.LabelsToRenderDataCalculatorOptions.max_num_labels)
  return _internal_max_num_labels();
}
inline void LabelsToRenderDataCalculatorOptions::_internal_set_max_num_labels(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000200u;
  max_num_labels_ = value;
}
inline void LabelsToRenderDataCalculatorOptions::set_max_num_labels(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_max_num_labels(value);
  // @@protoc_insertion_point(field_set:mediapipe.LabelsToRenderDataCalculatorOptions.max_num_labels)
}

// optional int32 font_face = 5 [default = 0];
inline bool LabelsToRenderDataCalculatorOptions::_internal_has_font_face() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool LabelsToRenderDataCalculatorOptions::has_font_face() const {
  return _internal_has_font_face();
}
inline void LabelsToRenderDataCalculatorOptions::clear_font_face() {
  font_face_ = 0;
  _has_bits_[0] &= ~0x00000001u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 LabelsToRenderDataCalculatorOptions::_internal_font_face() const {
  return font_face_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 LabelsToRenderDataCalculatorOptions::font_face() const {
  // @@protoc_insertion_point(field_get:mediapipe.LabelsToRenderDataCalculatorOptions.font_face)
  return _internal_font_face();
}
inline void LabelsToRenderDataCalculatorOptions::_internal_set_font_face(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000001u;
  font_face_ = value;
}
inline void LabelsToRenderDataCalculatorOptions::set_font_face(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_font_face(value);
  // @@protoc_insertion_point(field_set:mediapipe.LabelsToRenderDataCalculatorOptions.font_face)
}

// optional .mediapipe.LabelsToRenderDataCalculatorOptions.Location location = 6 [default = TOP_LEFT];
inline bool LabelsToRenderDataCalculatorOptions::_internal_has_location() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool LabelsToRenderDataCalculatorOptions::has_location() const {
  return _internal_has_location();
}
inline void LabelsToRenderDataCalculatorOptions::clear_location() {
  location_ = 0;
  _has_bits_[0] &= ~0x00000002u;
}
inline ::mediapipe::LabelsToRenderDataCalculatorOptions_Location LabelsToRenderDataCalculatorOptions::_internal_location() const {
  return static_cast< ::mediapipe::LabelsToRenderDataCalculatorOptions_Location >(location_);
}
inline ::mediapipe::LabelsToRenderDataCalculatorOptions_Location LabelsToRenderDataCalculatorOptions::location() const {
  // @@protoc_insertion_point(field_get:mediapipe.LabelsToRenderDataCalculatorOptions.location)
  return _internal_location();
}
inline void LabelsToRenderDataCalculatorOptions::_internal_set_location(::mediapipe::LabelsToRenderDataCalculatorOptions_Location value) {
  assert(::mediapipe::LabelsToRenderDataCalculatorOptions_Location_IsValid(value));
  _has_bits_[0] |= 0x00000002u;
  location_ = value;
}
inline void LabelsToRenderDataCalculatorOptions::set_location(::mediapipe::LabelsToRenderDataCalculatorOptions_Location value) {
  _internal_set_location(value);
  // @@protoc_insertion_point(field_set:mediapipe.LabelsToRenderDataCalculatorOptions.location)
}

// optional bool use_display_name = 9 [default = false];
inline bool LabelsToRenderDataCalculatorOptions::_internal_has_use_display_name() const {
  bool value = (_has_bits_[0] & 0x00000020u) != 0;
  return value;
}
inline bool LabelsToRenderDataCalculatorOptions::has_use_display_name() const {
  return _internal_has_use_display_name();
}
inline void LabelsToRenderDataCalculatorOptions::clear_use_display_name() {
  use_display_name_ = false;
  _has_bits_[0] &= ~0x00000020u;
}
inline bool LabelsToRenderDataCalculatorOptions::_internal_use_display_name() const {
  return use_display_name_;
}
inline bool LabelsToRenderDataCalculatorOptions::use_display_name() const {
  // @@protoc_insertion_point(field_get:mediapipe.LabelsToRenderDataCalculatorOptions.use_display_name)
  return _internal_use_display_name();
}
inline void LabelsToRenderDataCalculatorOptions::_internal_set_use_display_name(bool value) {
  _has_bits_[0] |= 0x00000020u;
  use_display_name_ = value;
}
inline void LabelsToRenderDataCalculatorOptions::set_use_display_name(bool value) {
  _internal_set_use_display_name(value);
  // @@protoc_insertion_point(field_set:mediapipe.LabelsToRenderDataCalculatorOptions.use_display_name)
}

// optional bool display_classification_score = 10 [default = false];
inline bool LabelsToRenderDataCalculatorOptions::_internal_has_display_classification_score() const {
  bool value = (_has_bits_[0] & 0x00000040u) != 0;
  return value;
}
inline bool LabelsToRenderDataCalculatorOptions::has_display_classification_score() const {
  return _internal_has_display_classification_score();
}
inline void LabelsToRenderDataCalculatorOptions::clear_display_classification_score() {
  display_classification_score_ = false;
  _has_bits_[0] &= ~0x00000040u;
}
inline bool LabelsToRenderDataCalculatorOptions::_internal_display_classification_score() const {
  return display_classification_score_;
}
inline bool LabelsToRenderDataCalculatorOptions::display_classification_score() const {
  // @@protoc_insertion_point(field_get:mediapipe.LabelsToRenderDataCalculatorOptions.display_classification_score)
  return _internal_display_classification_score();
}
inline void LabelsToRenderDataCalculatorOptions::_internal_set_display_classification_score(bool value) {
  _has_bits_[0] |= 0x00000040u;
  display_classification_score_ = value;
}
inline void LabelsToRenderDataCalculatorOptions::set_display_classification_score(bool value) {
  _internal_set_display_classification_score(value);
  // @@protoc_insertion_point(field_set:mediapipe.LabelsToRenderDataCalculatorOptions.display_classification_score)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

PROTOBUF_NAMESPACE_OPEN

template <> struct is_proto_enum< ::mediapipe::LabelsToRenderDataCalculatorOptions_Location> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::mediapipe::LabelsToRenderDataCalculatorOptions_Location>() {
  return ::mediapipe::LabelsToRenderDataCalculatorOptions_Location_descriptor();
}

PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)

#include <x/google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2futil_2flabels_5fto_5frender_5fdata_5fcalculator_2eproto
