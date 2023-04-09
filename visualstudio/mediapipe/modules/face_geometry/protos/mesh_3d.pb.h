// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/modules/face_geometry/protos/mesh_3d.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fmesh_5f3d_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fmesh_5f3d_2eproto

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
// @@protoc_insertion_point(includes)
#include <x/google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fmesh_5f3d_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fmesh_5f3d_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fmesh_5f3d_2eproto;
namespace mediapipe {
namespace face_geometry {
class Mesh3d;
struct Mesh3dDefaultTypeInternal;
extern Mesh3dDefaultTypeInternal _Mesh3d_default_instance_;
}  // namespace face_geometry
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::face_geometry::Mesh3d* Arena::CreateMaybeMessage<::mediapipe::face_geometry::Mesh3d>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {
namespace face_geometry {

enum Mesh3d_VertexType : int {
  Mesh3d_VertexType_VERTEX_PT = 0
};
bool Mesh3d_VertexType_IsValid(int value);
constexpr Mesh3d_VertexType Mesh3d_VertexType_VertexType_MIN = Mesh3d_VertexType_VERTEX_PT;
constexpr Mesh3d_VertexType Mesh3d_VertexType_VertexType_MAX = Mesh3d_VertexType_VERTEX_PT;
constexpr int Mesh3d_VertexType_VertexType_ARRAYSIZE = Mesh3d_VertexType_VertexType_MAX + 1;

const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* Mesh3d_VertexType_descriptor();
template<typename T>
inline const std::string& Mesh3d_VertexType_Name(T enum_t_value) {
  static_assert(::std::is_same<T, Mesh3d_VertexType>::value ||
    ::std::is_integral<T>::value,
    "Incorrect type passed to function Mesh3d_VertexType_Name.");
  return ::PROTOBUF_NAMESPACE_ID::internal::NameOfEnum(
    Mesh3d_VertexType_descriptor(), enum_t_value);
}
inline bool Mesh3d_VertexType_Parse(
    ::PROTOBUF_NAMESPACE_ID::ConstStringParam name, Mesh3d_VertexType* value) {
  return ::PROTOBUF_NAMESPACE_ID::internal::ParseNamedEnum<Mesh3d_VertexType>(
    Mesh3d_VertexType_descriptor(), name, value);
}
enum Mesh3d_PrimitiveType : int {
  Mesh3d_PrimitiveType_TRIANGLE = 0
};
bool Mesh3d_PrimitiveType_IsValid(int value);
constexpr Mesh3d_PrimitiveType Mesh3d_PrimitiveType_PrimitiveType_MIN = Mesh3d_PrimitiveType_TRIANGLE;
constexpr Mesh3d_PrimitiveType Mesh3d_PrimitiveType_PrimitiveType_MAX = Mesh3d_PrimitiveType_TRIANGLE;
constexpr int Mesh3d_PrimitiveType_PrimitiveType_ARRAYSIZE = Mesh3d_PrimitiveType_PrimitiveType_MAX + 1;

const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* Mesh3d_PrimitiveType_descriptor();
template<typename T>
inline const std::string& Mesh3d_PrimitiveType_Name(T enum_t_value) {
  static_assert(::std::is_same<T, Mesh3d_PrimitiveType>::value ||
    ::std::is_integral<T>::value,
    "Incorrect type passed to function Mesh3d_PrimitiveType_Name.");
  return ::PROTOBUF_NAMESPACE_ID::internal::NameOfEnum(
    Mesh3d_PrimitiveType_descriptor(), enum_t_value);
}
inline bool Mesh3d_PrimitiveType_Parse(
    ::PROTOBUF_NAMESPACE_ID::ConstStringParam name, Mesh3d_PrimitiveType* value) {
  return ::PROTOBUF_NAMESPACE_ID::internal::ParseNamedEnum<Mesh3d_PrimitiveType>(
    Mesh3d_PrimitiveType_descriptor(), name, value);
}
// ===================================================================

class Mesh3d PROTOBUF_FINAL :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.face_geometry.Mesh3d) */ {
 public:
  inline Mesh3d() : Mesh3d(nullptr) {}
  ~Mesh3d() override;
  explicit constexpr Mesh3d(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  Mesh3d(const Mesh3d& from);
  Mesh3d(Mesh3d&& from) noexcept
    : Mesh3d() {
    *this = ::std::move(from);
  }

  inline Mesh3d& operator=(const Mesh3d& from) {
    CopyFrom(from);
    return *this;
  }
  inline Mesh3d& operator=(Mesh3d&& from) noexcept {
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
  static const Mesh3d& default_instance() {
    return *internal_default_instance();
  }
  static inline const Mesh3d* internal_default_instance() {
    return reinterpret_cast<const Mesh3d*>(
               &_Mesh3d_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(Mesh3d& a, Mesh3d& b) {
    a.Swap(&b);
  }
  inline void Swap(Mesh3d* other) {
    if (other == this) return;
    if (GetArena() == other->GetArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(Mesh3d* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetArena() == other->GetArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline Mesh3d* New() const final {
    return CreateMaybeMessage<Mesh3d>(nullptr);
  }

  Mesh3d* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<Mesh3d>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const Mesh3d& from);
  void MergeFrom(const Mesh3d& from);
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
  void InternalSwap(Mesh3d* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.face_geometry.Mesh3d";
  }
  protected:
  explicit Mesh3d(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef Mesh3d_VertexType VertexType;
  static constexpr VertexType VERTEX_PT =
    Mesh3d_VertexType_VERTEX_PT;
  static inline bool VertexType_IsValid(int value) {
    return Mesh3d_VertexType_IsValid(value);
  }
  static constexpr VertexType VertexType_MIN =
    Mesh3d_VertexType_VertexType_MIN;
  static constexpr VertexType VertexType_MAX =
    Mesh3d_VertexType_VertexType_MAX;
  static constexpr int VertexType_ARRAYSIZE =
    Mesh3d_VertexType_VertexType_ARRAYSIZE;
  static inline const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor*
  VertexType_descriptor() {
    return Mesh3d_VertexType_descriptor();
  }
  template<typename T>
  static inline const std::string& VertexType_Name(T enum_t_value) {
    static_assert(::std::is_same<T, VertexType>::value ||
      ::std::is_integral<T>::value,
      "Incorrect type passed to function VertexType_Name.");
    return Mesh3d_VertexType_Name(enum_t_value);
  }
  static inline bool VertexType_Parse(::PROTOBUF_NAMESPACE_ID::ConstStringParam name,
      VertexType* value) {
    return Mesh3d_VertexType_Parse(name, value);
  }

  typedef Mesh3d_PrimitiveType PrimitiveType;
  static constexpr PrimitiveType TRIANGLE =
    Mesh3d_PrimitiveType_TRIANGLE;
  static inline bool PrimitiveType_IsValid(int value) {
    return Mesh3d_PrimitiveType_IsValid(value);
  }
  static constexpr PrimitiveType PrimitiveType_MIN =
    Mesh3d_PrimitiveType_PrimitiveType_MIN;
  static constexpr PrimitiveType PrimitiveType_MAX =
    Mesh3d_PrimitiveType_PrimitiveType_MAX;
  static constexpr int PrimitiveType_ARRAYSIZE =
    Mesh3d_PrimitiveType_PrimitiveType_ARRAYSIZE;
  static inline const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor*
  PrimitiveType_descriptor() {
    return Mesh3d_PrimitiveType_descriptor();
  }
  template<typename T>
  static inline const std::string& PrimitiveType_Name(T enum_t_value) {
    static_assert(::std::is_same<T, PrimitiveType>::value ||
      ::std::is_integral<T>::value,
      "Incorrect type passed to function PrimitiveType_Name.");
    return Mesh3d_PrimitiveType_Name(enum_t_value);
  }
  static inline bool PrimitiveType_Parse(::PROTOBUF_NAMESPACE_ID::ConstStringParam name,
      PrimitiveType* value) {
    return Mesh3d_PrimitiveType_Parse(name, value);
  }

  // accessors -------------------------------------------------------

  enum : int {
    kVertexBufferFieldNumber = 3,
    kIndexBufferFieldNumber = 4,
    kVertexTypeFieldNumber = 1,
    kPrimitiveTypeFieldNumber = 2,
  };
  // repeated float vertex_buffer = 3;
  int vertex_buffer_size() const;
  private:
  int _internal_vertex_buffer_size() const;
  public:
  void clear_vertex_buffer();
  private:
  float _internal_vertex_buffer(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      _internal_vertex_buffer() const;
  void _internal_add_vertex_buffer(float value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      _internal_mutable_vertex_buffer();
  public:
  float vertex_buffer(int index) const;
  void set_vertex_buffer(int index, float value);
  void add_vertex_buffer(float value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
      vertex_buffer() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
      mutable_vertex_buffer();

  // repeated uint32 index_buffer = 4;
  int index_buffer_size() const;
  private:
  int _internal_index_buffer_size() const;
  public:
  void clear_index_buffer();
  private:
  ::PROTOBUF_NAMESPACE_ID::uint32 _internal_index_buffer(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::uint32 >&
      _internal_index_buffer() const;
  void _internal_add_index_buffer(::PROTOBUF_NAMESPACE_ID::uint32 value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::uint32 >*
      _internal_mutable_index_buffer();
  public:
  ::PROTOBUF_NAMESPACE_ID::uint32 index_buffer(int index) const;
  void set_index_buffer(int index, ::PROTOBUF_NAMESPACE_ID::uint32 value);
  void add_index_buffer(::PROTOBUF_NAMESPACE_ID::uint32 value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::uint32 >&
      index_buffer() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::uint32 >*
      mutable_index_buffer();

  // optional .mediapipe.face_geometry.Mesh3d.VertexType vertex_type = 1;
  bool has_vertex_type() const;
  private:
  bool _internal_has_vertex_type() const;
  public:
  void clear_vertex_type();
  ::mediapipe::face_geometry::Mesh3d_VertexType vertex_type() const;
  void set_vertex_type(::mediapipe::face_geometry::Mesh3d_VertexType value);
  private:
  ::mediapipe::face_geometry::Mesh3d_VertexType _internal_vertex_type() const;
  void _internal_set_vertex_type(::mediapipe::face_geometry::Mesh3d_VertexType value);
  public:

  // optional .mediapipe.face_geometry.Mesh3d.PrimitiveType primitive_type = 2;
  bool has_primitive_type() const;
  private:
  bool _internal_has_primitive_type() const;
  public:
  void clear_primitive_type();
  ::mediapipe::face_geometry::Mesh3d_PrimitiveType primitive_type() const;
  void set_primitive_type(::mediapipe::face_geometry::Mesh3d_PrimitiveType value);
  private:
  ::mediapipe::face_geometry::Mesh3d_PrimitiveType _internal_primitive_type() const;
  void _internal_set_primitive_type(::mediapipe::face_geometry::Mesh3d_PrimitiveType value);
  public:

  // @@protoc_insertion_point(class_scope:mediapipe.face_geometry.Mesh3d)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< float > vertex_buffer_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::uint32 > index_buffer_;
  int vertex_type_;
  int primitive_type_;
  friend struct ::TableStruct_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fmesh_5f3d_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// Mesh3d

// optional .mediapipe.face_geometry.Mesh3d.VertexType vertex_type = 1;
inline bool Mesh3d::_internal_has_vertex_type() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool Mesh3d::has_vertex_type() const {
  return _internal_has_vertex_type();
}
inline void Mesh3d::clear_vertex_type() {
  vertex_type_ = 0;
  _has_bits_[0] &= ~0x00000001u;
}
inline ::mediapipe::face_geometry::Mesh3d_VertexType Mesh3d::_internal_vertex_type() const {
  return static_cast< ::mediapipe::face_geometry::Mesh3d_VertexType >(vertex_type_);
}
inline ::mediapipe::face_geometry::Mesh3d_VertexType Mesh3d::vertex_type() const {
  // @@protoc_insertion_point(field_get:mediapipe.face_geometry.Mesh3d.vertex_type)
  return _internal_vertex_type();
}
inline void Mesh3d::_internal_set_vertex_type(::mediapipe::face_geometry::Mesh3d_VertexType value) {
  assert(::mediapipe::face_geometry::Mesh3d_VertexType_IsValid(value));
  _has_bits_[0] |= 0x00000001u;
  vertex_type_ = value;
}
inline void Mesh3d::set_vertex_type(::mediapipe::face_geometry::Mesh3d_VertexType value) {
  _internal_set_vertex_type(value);
  // @@protoc_insertion_point(field_set:mediapipe.face_geometry.Mesh3d.vertex_type)
}

// optional .mediapipe.face_geometry.Mesh3d.PrimitiveType primitive_type = 2;
inline bool Mesh3d::_internal_has_primitive_type() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool Mesh3d::has_primitive_type() const {
  return _internal_has_primitive_type();
}
inline void Mesh3d::clear_primitive_type() {
  primitive_type_ = 0;
  _has_bits_[0] &= ~0x00000002u;
}
inline ::mediapipe::face_geometry::Mesh3d_PrimitiveType Mesh3d::_internal_primitive_type() const {
  return static_cast< ::mediapipe::face_geometry::Mesh3d_PrimitiveType >(primitive_type_);
}
inline ::mediapipe::face_geometry::Mesh3d_PrimitiveType Mesh3d::primitive_type() const {
  // @@protoc_insertion_point(field_get:mediapipe.face_geometry.Mesh3d.primitive_type)
  return _internal_primitive_type();
}
inline void Mesh3d::_internal_set_primitive_type(::mediapipe::face_geometry::Mesh3d_PrimitiveType value) {
  assert(::mediapipe::face_geometry::Mesh3d_PrimitiveType_IsValid(value));
  _has_bits_[0] |= 0x00000002u;
  primitive_type_ = value;
}
inline void Mesh3d::set_primitive_type(::mediapipe::face_geometry::Mesh3d_PrimitiveType value) {
  _internal_set_primitive_type(value);
  // @@protoc_insertion_point(field_set:mediapipe.face_geometry.Mesh3d.primitive_type)
}

// repeated float vertex_buffer = 3;
inline int Mesh3d::_internal_vertex_buffer_size() const {
  return vertex_buffer_.size();
}
inline int Mesh3d::vertex_buffer_size() const {
  return _internal_vertex_buffer_size();
}
inline void Mesh3d::clear_vertex_buffer() {
  vertex_buffer_.Clear();
}
inline float Mesh3d::_internal_vertex_buffer(int index) const {
  return vertex_buffer_.Get(index);
}
inline float Mesh3d::vertex_buffer(int index) const {
  // @@protoc_insertion_point(field_get:mediapipe.face_geometry.Mesh3d.vertex_buffer)
  return _internal_vertex_buffer(index);
}
inline void Mesh3d::set_vertex_buffer(int index, float value) {
  vertex_buffer_.Set(index, value);
  // @@protoc_insertion_point(field_set:mediapipe.face_geometry.Mesh3d.vertex_buffer)
}
inline void Mesh3d::_internal_add_vertex_buffer(float value) {
  vertex_buffer_.Add(value);
}
inline void Mesh3d::add_vertex_buffer(float value) {
  _internal_add_vertex_buffer(value);
  // @@protoc_insertion_point(field_add:mediapipe.face_geometry.Mesh3d.vertex_buffer)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
Mesh3d::_internal_vertex_buffer() const {
  return vertex_buffer_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >&
Mesh3d::vertex_buffer() const {
  // @@protoc_insertion_point(field_list:mediapipe.face_geometry.Mesh3d.vertex_buffer)
  return _internal_vertex_buffer();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
Mesh3d::_internal_mutable_vertex_buffer() {
  return &vertex_buffer_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< float >*
Mesh3d::mutable_vertex_buffer() {
  // @@protoc_insertion_point(field_mutable_list:mediapipe.face_geometry.Mesh3d.vertex_buffer)
  return _internal_mutable_vertex_buffer();
}

// repeated uint32 index_buffer = 4;
inline int Mesh3d::_internal_index_buffer_size() const {
  return index_buffer_.size();
}
inline int Mesh3d::index_buffer_size() const {
  return _internal_index_buffer_size();
}
inline void Mesh3d::clear_index_buffer() {
  index_buffer_.Clear();
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 Mesh3d::_internal_index_buffer(int index) const {
  return index_buffer_.Get(index);
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 Mesh3d::index_buffer(int index) const {
  // @@protoc_insertion_point(field_get:mediapipe.face_geometry.Mesh3d.index_buffer)
  return _internal_index_buffer(index);
}
inline void Mesh3d::set_index_buffer(int index, ::PROTOBUF_NAMESPACE_ID::uint32 value) {
  index_buffer_.Set(index, value);
  // @@protoc_insertion_point(field_set:mediapipe.face_geometry.Mesh3d.index_buffer)
}
inline void Mesh3d::_internal_add_index_buffer(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  index_buffer_.Add(value);
}
inline void Mesh3d::add_index_buffer(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  _internal_add_index_buffer(value);
  // @@protoc_insertion_point(field_add:mediapipe.face_geometry.Mesh3d.index_buffer)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::uint32 >&
Mesh3d::_internal_index_buffer() const {
  return index_buffer_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::uint32 >&
Mesh3d::index_buffer() const {
  // @@protoc_insertion_point(field_list:mediapipe.face_geometry.Mesh3d.index_buffer)
  return _internal_index_buffer();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::uint32 >*
Mesh3d::_internal_mutable_index_buffer() {
  return &index_buffer_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::uint32 >*
Mesh3d::mutable_index_buffer() {
  // @@protoc_insertion_point(field_mutable_list:mediapipe.face_geometry.Mesh3d.index_buffer)
  return _internal_mutable_index_buffer();
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace face_geometry
}  // namespace mediapipe

PROTOBUF_NAMESPACE_OPEN

template <> struct is_proto_enum< ::mediapipe::face_geometry::Mesh3d_VertexType> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::mediapipe::face_geometry::Mesh3d_VertexType>() {
  return ::mediapipe::face_geometry::Mesh3d_VertexType_descriptor();
}
template <> struct is_proto_enum< ::mediapipe::face_geometry::Mesh3d_PrimitiveType> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::mediapipe::face_geometry::Mesh3d_PrimitiveType>() {
  return ::mediapipe::face_geometry::Mesh3d_PrimitiveType_descriptor();
}

PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)

#include <x/google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fmesh_5f3d_2eproto
