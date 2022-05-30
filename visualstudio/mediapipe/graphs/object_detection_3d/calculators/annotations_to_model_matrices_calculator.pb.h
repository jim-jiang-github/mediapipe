// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/graphs/object_detection_3d/calculators/annotations_to_model_matrices_calculator.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fgraphs_2fobject_5fdetection_5f3d_2fcalculators_2fannotations_5fto_5fmodel_5fmatrices_5fcalculator_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fgraphs_2fobject_5fdetection_5f3d_2fcalculators_2fannotations_5fto_5fmodel_5fmatrices_5fcalculator_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
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

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
#include "mediapipe/framework/calculator.pb.h"
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fgraphs_2fobject_5fdetection_5f3d_2fcalculators_2fannotations_5fto_5fmodel_5fmatrices_5fcalculator_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fgraphs_2fobject_5fdetection_5f3d_2fcalculators_2fannotations_5fto_5fmodel_5fmatrices_5fcalculator_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fgraphs_2fobject_5fdetection_5f3d_2fcalculators_2fannotations_5fto_5fmodel_5fmatrices_5fcalculator_2eproto;
namespace mediapipe {
class AnnotationsToModelMatricesCalculatorOptions;
struct AnnotationsToModelMatricesCalculatorOptionsDefaultTypeInternal;
extern AnnotationsToModelMatricesCalculatorOptionsDefaultTypeInternal _AnnotationsToModelMatricesCalculatorOptions_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::AnnotationsToModelMatricesCalculatorOptions* Arena::CreateMaybeMessage<::mediapipe::AnnotationsToModelMatricesCalculatorOptions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class AnnotationsToModelMatricesCalculatorOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.AnnotationsToModelMatricesCalculatorOptions) */ {
 public:
  inline AnnotationsToModelMatricesCalculatorOptions() : AnnotationsToModelMatricesCalculatorOptions(nullptr) {}
  ~AnnotationsToModelMatricesCalculatorOptions() override;
  explicit constexpr AnnotationsToModelMatricesCalculatorOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  AnnotationsToModelMatricesCalculatorOptions(const AnnotationsToModelMatricesCalculatorOptions& from);
  AnnotationsToModelMatricesCalculatorOptions(AnnotationsToModelMatricesCalculatorOptions&& from) noexcept
    : AnnotationsToModelMatricesCalculatorOptions() {
    *this = ::std::move(from);
  }

  inline AnnotationsToModelMatricesCalculatorOptions& operator=(const AnnotationsToModelMatricesCalculatorOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline AnnotationsToModelMatricesCalculatorOptions& operator=(AnnotationsToModelMatricesCalculatorOptions&& from) noexcept {
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
  static const AnnotationsToModelMatricesCalculatorOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const AnnotationsToModelMatricesCalculatorOptions* internal_default_instance() {
    return reinterpret_cast<const AnnotationsToModelMatricesCalculatorOptions*>(
               &_AnnotationsToModelMatricesCalculatorOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(AnnotationsToModelMatricesCalculatorOptions& a, AnnotationsToModelMatricesCalculatorOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(AnnotationsToModelMatricesCalculatorOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(AnnotationsToModelMatricesCalculatorOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline AnnotationsToModelMatricesCalculatorOptions* New() const final {
    return CreateMaybeMessage<AnnotationsToModelMatricesCalculatorOptions>(nullptr);
  }

  AnnotationsToModelMatricesCalculatorOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<AnnotationsToModelMatricesCalculatorOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const AnnotationsToModelMatricesCalculatorOptions& from);
  void MergeFrom(const AnnotationsToModelMatricesCalculatorOptions& from);
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
  void InternalSwap(AnnotationsToModelMatricesCalculatorOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.AnnotationsToModelMatricesCalculatorOptions";
  }
  protected:
  explicit AnnotationsToModelMatricesCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kModelScaleFieldNumber = 1,
    kModelTransformationFieldNumber = 2,
  };
  // repeated float model_scale = 1;
  int model_scale_size() const;
  private:
  int _internal_model_scale_size() const;
  public:
  void clear_model_scale();
  private:
  float _internal_model_scale(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      _internal_model_scale() const;
  void _internal_add_model_scale(float value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      _internal_mutable_model_scale();
  public:
  float model_scale(int index) const;
  void set_model_scale(int index, float value);
  void add_model_scale(float value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      model_scale() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      mutable_model_scale();

  // repeated float model_transformation = 2;
  int model_transformation_size() const;
  private:
  int _internal_model_transformation_size() const;
  public:
  void clear_model_transformation();
  private:
  float _internal_model_transformation(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      _internal_model_transformation() const;
  void _internal_add_model_transformation(float value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      _internal_mutable_model_transformation();
  public:
  float model_transformation(int index) const;
  void set_model_transformation(int index, float value);
  void add_model_transformation(float value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      model_transformation() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      mutable_model_transformation();

  static const int kExtFieldNumber = 290166283;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::AnnotationsToModelMatricesCalculatorOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.AnnotationsToModelMatricesCalculatorOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float > model_scale_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float > model_transformation_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_mediapipe_2fgraphs_2fobject_5fdetection_5f3d_2fcalculators_2fannotations_5fto_5fmodel_5fmatrices_5fcalculator_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// AnnotationsToModelMatricesCalculatorOptions

// repeated float model_scale = 1;
inline int AnnotationsToModelMatricesCalculatorOptions::_internal_model_scale_size() const {
  return model_scale_.size();
}
inline int AnnotationsToModelMatricesCalculatorOptions::model_scale_size() const {
  return _internal_model_scale_size();
}
inline void AnnotationsToModelMatricesCalculatorOptions::clear_model_scale() {
  model_scale_.Clear();
}
inline float AnnotationsToModelMatricesCalculatorOptions::_internal_model_scale(int index) const {
  return model_scale_.Get(index);
}
inline float AnnotationsToModelMatricesCalculatorOptions::model_scale(int index) const {
  // @@protoc_insertion_point(field_get:mediapipe.AnnotationsToModelMatricesCalculatorOptions.model_scale)
  return _internal_model_scale(index);
}
inline void AnnotationsToModelMatricesCalculatorOptions::set_model_scale(int index, float value) {
  model_scale_.Set(index, value);
  // @@protoc_insertion_point(field_set:mediapipe.AnnotationsToModelMatricesCalculatorOptions.model_scale)
}
inline void AnnotationsToModelMatricesCalculatorOptions::_internal_add_model_scale(float value) {
  model_scale_.Add(value);
}
inline void AnnotationsToModelMatricesCalculatorOptions::add_model_scale(float value) {
  _internal_add_model_scale(value);
  // @@protoc_insertion_point(field_add:mediapipe.AnnotationsToModelMatricesCalculatorOptions.model_scale)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
AnnotationsToModelMatricesCalculatorOptions::_internal_model_scale() const {
  return model_scale_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
AnnotationsToModelMatricesCalculatorOptions::model_scale() const {
  // @@protoc_insertion_point(field_list:mediapipe.AnnotationsToModelMatricesCalculatorOptions.model_scale)
  return _internal_model_scale();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
AnnotationsToModelMatricesCalculatorOptions::_internal_mutable_model_scale() {
  return &model_scale_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
AnnotationsToModelMatricesCalculatorOptions::mutable_model_scale() {
  // @@protoc_insertion_point(field_mutable_list:mediapipe.AnnotationsToModelMatricesCalculatorOptions.model_scale)
  return _internal_mutable_model_scale();
}

// repeated float model_transformation = 2;
inline int AnnotationsToModelMatricesCalculatorOptions::_internal_model_transformation_size() const {
  return model_transformation_.size();
}
inline int AnnotationsToModelMatricesCalculatorOptions::model_transformation_size() const {
  return _internal_model_transformation_size();
}
inline void AnnotationsToModelMatricesCalculatorOptions::clear_model_transformation() {
  model_transformation_.Clear();
}
inline float AnnotationsToModelMatricesCalculatorOptions::_internal_model_transformation(int index) const {
  return model_transformation_.Get(index);
}
inline float AnnotationsToModelMatricesCalculatorOptions::model_transformation(int index) const {
  // @@protoc_insertion_point(field_get:mediapipe.AnnotationsToModelMatricesCalculatorOptions.model_transformation)
  return _internal_model_transformation(index);
}
inline void AnnotationsToModelMatricesCalculatorOptions::set_model_transformation(int index, float value) {
  model_transformation_.Set(index, value);
  // @@protoc_insertion_point(field_set:mediapipe.AnnotationsToModelMatricesCalculatorOptions.model_transformation)
}
inline void AnnotationsToModelMatricesCalculatorOptions::_internal_add_model_transformation(float value) {
  model_transformation_.Add(value);
}
inline void AnnotationsToModelMatricesCalculatorOptions::add_model_transformation(float value) {
  _internal_add_model_transformation(value);
  // @@protoc_insertion_point(field_add:mediapipe.AnnotationsToModelMatricesCalculatorOptions.model_transformation)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
AnnotationsToModelMatricesCalculatorOptions::_internal_model_transformation() const {
  return model_transformation_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
AnnotationsToModelMatricesCalculatorOptions::model_transformation() const {
  // @@protoc_insertion_point(field_list:mediapipe.AnnotationsToModelMatricesCalculatorOptions.model_transformation)
  return _internal_model_transformation();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
AnnotationsToModelMatricesCalculatorOptions::_internal_mutable_model_transformation() {
  return &model_transformation_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
AnnotationsToModelMatricesCalculatorOptions::mutable_model_transformation() {
  // @@protoc_insertion_point(field_mutable_list:mediapipe.AnnotationsToModelMatricesCalculatorOptions.model_transformation)
  return _internal_mutable_model_transformation();
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fgraphs_2fobject_5fdetection_5f3d_2fcalculators_2fannotations_5fto_5fmodel_5fmatrices_5fcalculator_2eproto
