#if __cplusplus >= 201703L

#ifdef _WIN32
//
// error C4996: 'std::iterator<std::forward_iterator_tag,T,ptrdiff_t,T *,T &>':
// warning STL4015: The std::iterator class template (used as a base class to
// provide typedefs) is deprecated in C++17. (The <iterator> header is NOT
// deprecated.) The C++ Standard has never required user-defined iterators to
// derive from std::iterator. Stop deriving from std::iterator and
// start providing publicly accessible typedefs named iterator_category,
// value_type, difference_type, pointer, and reference. Note that value_type
// is required to be non-const, even for constant iterators.
//
// or do this in project Properties:
//   C/C++ -> Advanced -> Disable Specific Warnings -> 4996 
#pragma warning(disable:4996)
#endif

#endif

#include "mediapipe/framework/tool/options_registry.h"

#include "mediapipe/modules/face_detection/face_detection.pb.h"
#include "mediapipe/calculators/core/gate_calculator.pb.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"
#include "mediapipe/calculators/tflite/ssd_anchors_calculator.pb.h"

namespace {

class FileDescriptorSetBuilder {
  google::protobuf::FileDescriptorSet fd_set_;
  std::set<std::string> contains_;

  bool AddFileDescriptor_(google::protobuf::FileDescriptor const* fd) {
    auto const& name = fd->name();
    if (contains_.end()==contains_.find(name)) {
      google::protobuf::FileDescriptorProto* fdp = fd_set_.add_file();
      if (fdp) {
        fd->CopyTo(fdp);
        contains_.insert(name);
        return true;
      } 
      return false;
    }
    return true;
  }

public:
  template<typename T>
  bool Add(bool dependency=true) {
    return AddFileDescriptor(T::descriptor()->file(), dependency);
  }

  //
  // fd = T::descriptor()->file()
  bool AddFileDescriptor(google::protobuf::FileDescriptor const* fd, bool dependency) {
    if (dependency) {
      int const deps = fd->dependency_count();
      for (int i=0; i<deps; ++i) {
        if (!AddFileDescriptor(fd->dependency(i), dependency)) {
          return false;
        }
      }
    }

    return AddFileDescriptor_(fd);
  }

  bool SerializeAsString(std::string* ss) const {
    if (ss && fd_set_.file_size()>0) {
      *ss = fd_set_.SerializeAsString();
      return true;
    }
    return false;
  }
};

template<typename T>
void register_option_data(mediapipe::FieldData const& result) {
  // ensure this message's reflection is linked into the library...
  mediapipe::proto_ns::LinkMessageReflection<T>();

  // register
  mediapipe::tool::OptionsRegistry::Register(result);
}

}

//
// Started from v0.8.10.2, mediapipe framework requires client code to register
// 'options' before initializing calculatror graph. By design, the mediapipe
// bazel build outputs few intermediate files to be used for the final build stage.
// (e.g. #include notorious .inc files to define C++ string laterials)
//
// It's just, to me, over complicated(, ugly) and really pushes people
// away from mediapipe + bazel.
//
// Since all proto messagess are well defined in the code, it should be
// able to collect necessary proto data for registration during runtime.
// Anyway, Keep it simple, man!
//
// See below for details data you need to register 'options'....
//
inline void register_face_detection_options() {
  FileDescriptorSetBuilder fds_builder;
  fds_builder.Add<mediapipe::FaceDetectionOptions>(true);

//fds_builder.Add<mediapipe::GateCalculatorOptions>(false);
  fds_builder.Add<mediapipe::ImageToTensorCalculatorOptions>(false);
//fds_builder.Add<mediapipe::InferenceCalculatorOptions>(false);
  fds_builder.Add<mediapipe::TensorsToDetectionsCalculatorOptions>(false);
  fds_builder.Add<mediapipe::SsdAnchorsCalculatorOptions>(false);
//fds_builder.Add<mediapipe::GpuOrigin>(false);

  mediapipe::FieldData data;
  auto* message = data.mutable_message_value();
  if (fds_builder.SerializeAsString(message->mutable_value())) {
    *(message->mutable_type_url()) = "proto2.FileDescriptorSet";
    register_option_data<mediapipe::FaceDetectionOptions>(data);
  }
}

//
// bazel build detail explaination...
//
// In the process of face_detection bazel build, 'face_detection_proto' module
// caught my eyes... //mediapipe/modules/face_detection/BUILD (line 113)
//
// mediapipe_proto_library(
//    name = "face_detection_proto",
//    srcs = ["face_detection.proto"],
//    deps = [
//        "//mediapipe/calculators/core:gate_calculator_proto",
//        "//mediapipe/calculators/tensor:image_to_tensor_calculator_proto",
//        "//mediapipe/calculators/tensor:inference_calculator_proto",
//        "//mediapipe/calculators/tensor:tensors_to_detections_calculator_proto",
//        "//mediapipe/calculators/tflite:ssd_anchors_calculator_proto",
//        "//mediapipe/framework:calculator_options_proto",
//        "//mediapipe/gpu:gpu_origin_proto",
//    ],
// )
//
// Let's check out the build rule mediapipe_proto_library(), which is
// defined in //mediapipe/framework/port/build_config.bzl (line 35)...
// def mediapipe_proto_library(
//          name,
//          srcs,
//          deps = [],
//          ......
//          ......
//          ......
//          def_options_lib = True,
//          ......):
//   ......
//   ......
//   ......
//   if def_options_lib:
//      cc_deps = replace_deps(deps, "_proto", "_cc_proto")
//      mediapipe_options_library(**provided_args(
//          name = replace_suffix(name, "_proto", "_options_lib"),
//          proto_lib = name,
//          deps = cc_deps,
//          visibility = visibility,
//          testonly = testonly,
//          compatible_with = compatible_with,
//      ))
//
//
// Again, mediapipe_options_library() is triggered. its definition can be found
// in //mediapipe/framework/tool/mediapipe_graph.bzl (line 205)...
// def mediapipe_options_library(
//        name,
//        proto_lib,
//        deps = [],
//        visibility = None,
//        testonly = None,
//        **kwargs):
//
//   transitive_descriptor_set(
//        name = proto_lib + "_transitive",
//        deps = [proto_lib],
//        testonly = testonly,
//    )
//    ......
//    data_as_c_string(
//        name = name + "_inc",
//        srcs = [proto_lib + "_transitive-transitive-descriptor-set.proto.bin"],
//        outs = [proto_lib + "_descriptors.inc"],
//    )
//    ......
//    ......
//
// transitive_descriptor_set() is in //mediapipe/framework/deps/descriptor_set.bzl
// and it serializes FileDescriptorSet containing proto_lib with all
// 'transitive dependencies' into a binary file named
// face_detection_proto_transitive-transitive-descriptor-set.proto.bin.
//
// next, data_as_c_string() converts the binary file into a bad-taste text file,
// face_detection_proto_descriptors.inc. (same usage as subgraphs).
//
// you can check both files out using notepad++, or run through below test function
// to have better understanding.
//
constexpr char const* transitive_descriptor_set =
    "../face_detection/face_detection_proto_transitive-transitive-descriptor-set.proto.bin";

// taken from //mediapipe/framework/tool/options_lib_template.cc
constexpr char kDescriptorContents[] =
#include "../face_detection/face_detection_proto_descriptors.inc"
    ;  // NOLINT(whitespace/semicolon)

bool test_transitive_descriptor_set() {
  printf("\ntest transitive-transitive-descriptor-set and face_detection_proto_descriptors.inc from bazel build\n");

  struct buffer_raii {
    uint8_t* desc{nullptr};
    int desc_len{0};
    ~buffer_raii() {
      if (desc) free(desc);
    }
  } buf;

  FILE* file = fopen(transitive_descriptor_set, "rb");
  if (file) {
    fseek(file, 0, SEEK_END);
    int const file_len = (int) ftell(file);
    buf.desc = (uint8_t*) malloc(file_len);
    if (buf.desc) {
      rewind(file);
      if (file_len==fread(buf.desc, 1, file_len, file)) {
        buf.desc_len = file_len;
      }
    }

    fclose(file);
  }

  // face_detection_proto_transitive-transitive-descriptor-set.proto.bin
  //          vs. face_detection_proto_descriptors.inc
  if (!buf.desc || (buf.desc_len!=((int) sizeof(kDescriptorContents)-1)) ||
      0!=memcmp(kDescriptorContents, buf.desc, buf.desc_len)) {
    printf("=> failed - transitive-descriptor-set file not load?\n\n");
    return false;
  }

  google::protobuf::FileDescriptorSet descriptor_set;
  if (!descriptor_set.ParseFromArray(buf.desc, buf.desc_len)) {
    printf("=> failed - !descriptor_set.ParseFromArray()?\n\n");
    return false;
  }

  std::string desc_str = descriptor_set.SerializeAsString();
  if (buf.desc_len!=desc_str.length() || 0!=memcmp(buf.desc, desc_str.data(), buf.desc_len) ||
      buf.desc_len!=descriptor_set.GetCachedSize() || buf.desc_len!=descriptor_set.ByteSizeLong()) {
     printf("=> failed - descriptor_set serialize data not match!\n\n");
    return false;
  }

  uint8_t const* ptr = buf.desc;
  uint8_t const* ptr_end = buf.desc + buf.desc_len;
  int files_ok = 0;
  for (google::protobuf::FileDescriptorProto const& file : descriptor_set.file()) {
    printf("  #%02d %s...", files_ok, file.name().c_str());

    // to enumerate all messages within...
  //for (auto& descp: file.message_type()) { printf(" %s", descp.name().c_str()); }

    // 10 = GOOGLE_PROTOBUF_WIRE_FORMAT_MAKE_TAG(1, WIRETYPE_LENGTH_DELIMITED);
    if (10!=*ptr) {
      printf(" magic failed!\n");
      break;
    }
    ++ptr;

    int decode_cache_size = *ptr;
    for (int n=0x80; 0x80&(*ptr); n<<=7) {
      uint8_t byte = *++ptr;
      assert(byte>0);
      if (byte>1) {
        decode_cache_size += (((int)byte)-1)*n;
      }
    }
    ++ptr;

    int const file_cached_size = file.GetCachedSize();
    if (decode_cache_size!=file_cached_size) {
      printf(" cached size failed!\n");
      break;
    }

    std::string ss = file.SerializeAsString();
    uint8_t const* ss_data = (uint8_t*) ss.c_str();
    int const ss_len = (int) ss.size();
    if (ss_len!=file_cached_size) {
      printf(" serialize data size failed!\n");
      break;
    }

    if ((ptr+ss_len)>ptr_end) {
      printf(" run out of desc data!\n");
      break;
    }

    if (memcmp(ptr, ss_data, ss_len)) {
      printf(" fiel serialize data not match!\n");
      break;
    }

    ptr += ss_len;
    ++files_ok;

    printf(" [OK] (%d bytes)\n", ss_len);
  }

  if (files_ok==descriptor_set.file_size() && ptr_end==ptr) {
    printf("=> OK!\n\n");
    return true;
  }

  printf("=> FAILED!\n\n");
  return false;
}