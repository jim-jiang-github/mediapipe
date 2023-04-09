// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/calculators/tflite/tflite_custom_op_resolver_calculator.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2ftflite_2ftflite_5fcustom_5fop_5fresolver_5fcalculator_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2ftflite_2ftflite_5fcustom_5fop_5fresolver_5fcalculator_2eproto

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
#include <x/google/protobuf/unknown_field_set.h>
#include "mediapipe/framework/calculator.pb.h"
// @@protoc_insertion_point(includes)
#include <x/google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fcalculators_2ftflite_2ftflite_5fcustom_5fop_5fresolver_5fcalculator_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fcalculators_2ftflite_2ftflite_5fcustom_5fop_5fresolver_5fcalculator_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fcalculators_2ftflite_2ftflite_5fcustom_5fop_5fresolver_5fcalculator_2eproto;
namespace mediapipe {
class TfLiteCustomOpResolverCalculatorOptions;
struct TfLiteCustomOpResolverCalculatorOptionsDefaultTypeInternal;
extern TfLiteCustomOpResolverCalculatorOptionsDefaultTypeInternal _TfLiteCustomOpResolverCalculatorOptions_default_instance_;
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::TfLiteCustomOpResolverCalculatorOptions* Arena::CreateMaybeMessage<::mediapipe::TfLiteCustomOpResolverCalculatorOptions>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {

// ===================================================================

class TfLiteCustomOpResolverCalculatorOptions PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.TfLiteCustomOpResolverCalculatorOptions) */ {
 public:
  inline TfLiteCustomOpResolverCalculatorOptions() : TfLiteCustomOpResolverCalculatorOptions(nullptr) {}
  ~TfLiteCustomOpResolverCalculatorOptions() override;
  explicit constexpr TfLiteCustomOpResolverCalculatorOptions(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  TfLiteCustomOpResolverCalculatorOptions(const TfLiteCustomOpResolverCalculatorOptions& from);
  TfLiteCustomOpResolverCalculatorOptions(TfLiteCustomOpResolverCalculatorOptions&& from) noexcept
    : TfLiteCustomOpResolverCalculatorOptions() {
    *this = ::std::move(from);
  }

  inline TfLiteCustomOpResolverCalculatorOptions& operator=(const TfLiteCustomOpResolverCalculatorOptions& from) {
    CopyFrom(from);
    return *this;
  }
  inline TfLiteCustomOpResolverCalculatorOptions& operator=(TfLiteCustomOpResolverCalculatorOptions&& from) noexcept {
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
  static const TfLiteCustomOpResolverCalculatorOptions& default_instance() {
    return *internal_default_instance();
  }
  static inline const TfLiteCustomOpResolverCalculatorOptions* internal_default_instance() {
    return reinterpret_cast<const TfLiteCustomOpResolverCalculatorOptions*>(
               &_TfLiteCustomOpResolverCalculatorOptions_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(TfLiteCustomOpResolverCalculatorOptions& a, TfLiteCustomOpResolverCalculatorOptions& b) {
    a.Swap(&b);
  }
  inline void Swap(TfLiteCustomOpResolverCalculatorOptions* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(TfLiteCustomOpResolverCalculatorOptions* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline TfLiteCustomOpResolverCalculatorOptions* New() const final {
    return CreateMaybeMessage<TfLiteCustomOpResolverCalculatorOptions>(nullptr);
  }

  TfLiteCustomOpResolverCalculatorOptions* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<TfLiteCustomOpResolverCalculatorOptions>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const TfLiteCustomOpResolverCalculatorOptions& from);
  void MergeFrom(const TfLiteCustomOpResolverCalculatorOptions& from);
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
  void InternalSwap(TfLiteCustomOpResolverCalculatorOptions* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.TfLiteCustomOpResolverCalculatorOptions";
  }
  protected:
  explicit TfLiteCustomOpResolverCalculatorOptions(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kUseGpuFieldNumber = 1,
  };
  // optional bool use_gpu = 1 [default = false];
  bool has_use_gpu() const;
  private:
  bool _internal_has_use_gpu() const;
  public:
  void clear_use_gpu();
  bool use_gpu() const;
  void set_use_gpu(bool value);
  private:
  bool _internal_use_gpu() const;
  void _internal_set_use_gpu(bool value);
  public:

  static const int kExtFieldNumber = 252087553;
  static ::PROTOBUF_NAMESPACE_ID::internal::ExtensionIdentifier< ::mediapipe::CalculatorOptions,
      ::PROTOBUF_NAMESPACE_ID::internal::MessageTypeTraits< ::mediapipe::TfLiteCustomOpResolverCalculatorOptions >, 11, false >
    ext;
  // @@protoc_insertion_point(class_scope:mediapipe.TfLiteCustomOpResolverCalculatorOptions)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  bool use_gpu_;
  friend struct ::TableStruct_mediapipe_2fcalculators_2ftflite_2ftflite_5fcustom_5fop_5fresolver_5fcalculator_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// TfLiteCustomOpResolverCalculatorOptions

// optional bool use_gpu = 1 [default = false];
inline bool TfLiteCustomOpResolverCalculatorOptions::_internal_has_use_gpu() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool TfLiteCustomOpResolverCalculatorOptions::has_use_gpu() const {
  return _internal_has_use_gpu();
}
inline void TfLiteCustomOpResolverCalculatorOptions::clear_use_gpu() {
  use_gpu_ = false;
  _has_bits_[0] &= ~0x00000001u;
}
inline bool TfLiteCustomOpResolverCalculatorOptions::_internal_use_gpu() const {
  return use_gpu_;
}
inline bool TfLiteCustomOpResolverCalculatorOptions::use_gpu() const {
  // @@protoc_insertion_point(field_get:mediapipe.TfLiteCustomOpResolverCalculatorOptions.use_gpu)
  return _internal_use_gpu();
}
inline void TfLiteCustomOpResolverCalculatorOptions::_internal_set_use_gpu(bool value) {
  _has_bits_[0] |= 0x00000001u;
  use_gpu_ = value;
}
inline void TfLiteCustomOpResolverCalculatorOptions::set_use_gpu(bool value) {
  _internal_set_use_gpu(value);
  // @@protoc_insertion_point(field_set:mediapipe.TfLiteCustomOpResolverCalculatorOptions.use_gpu)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace mediapipe

// @@protoc_insertion_point(global_scope)

#include <x/google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fcalculators_2ftflite_2ftflite_5fcustom_5fop_5fresolver_5fcalculator_2eproto
