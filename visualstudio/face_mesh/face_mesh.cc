#include "../calculator_graph_util.h"

// define graph loading function
DEFINE_LOAD_GRAPH("../../mediapipe/graphs/face_mesh/face_mesh_desktop_live.pbtxt")

namespace mediapipe {

#if !MEDIAPIPE_DISABLE_GPU
DEFINE_SUBGRAPH(FaceLandmarkFrontGpu, "../../mediapipe/modules/face_landmark/face_landmark_front_gpu.pbtxt");
DEFINE_SUBGRAPH(FaceLandmarkGpu, "../../mediapipe/modules/face_landmark/face_landmark_gpu.pbtxt");
DEFINE_SUBGRAPH(FaceDetectionFrontGpu, "face_detection_short_range_gpu.pbtxt");
DEFINE_SUBGRAPH(FaceRendererGpu, "../../mediapipe/graphs/face_mesh/subgraphs/face_renderer_gpu.pbtxt");
#endif

DEFINE_SUBGRAPH(FaceLandmarkFrontCpu, "../../mediapipe/modules/face_landmark/face_landmark_front_cpu.pbtxt");
  DEFINE_SUBGRAPH(FaceDetectionShortRangeCpu, "../face_detection/face_detection_short_range_cpu.pbtxt"); // ../../mediapipe/modules/face_detection
    DEFINE_SUBGRAPH(FaceDetectionShortRangeCommon, "../../mediapipe/modules/face_detection/face_detection_short_range_common.pbtxt");
  DEFINE_SUBGRAPH(FaceDetectionFrontDetectionToRoi, "../../mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt");
  DEFINE_SUBGRAPH(FaceLandmarkCpu, "../../mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt");
    DEFINE_SUBGRAPH(FaceLandmarksModelLoader, "face_landmarks_model_loader.pbtxt"); // ../../mediapipe/modules/face_landmark
    DEFINE_SUBGRAPH(TensorsToFaceLandmarks, "../../mediapipe/modules/face_landmark/tensors_to_face_landmarks.pbtxt");
    DEFINE_SUBGRAPH(TensorsToFaceLandmarksWithAttention, "../../mediapipe/modules/face_landmark/tensors_to_face_landmarks_with_attention.pbtxt");
  DEFINE_SUBGRAPH(FaceLandmarkLandmarksToRoi, "../../mediapipe/modules/face_landmark/face_landmark_landmarks_to_roi.pbtxt");
DEFINE_SUBGRAPH(FaceRendererCpu, "../../mediapipe/graphs/face_mesh/subgraphs/face_renderer_cpu.pbtxt");

#if 0
// quick debug/test
struct FaceLandmarksSmoothing : public Subgraph {
  absl::StatusOr<CalculatorGraphConfig> GetConfig(SubgraphOptions const&) override {
    CalculatorGraphConfig config;
    if (read_config_from_pbtxt(config, "face_landmarks_smoothing.pbtxt")) {
      return config;
    }
    return absl::InternalError("Could not parse subgraph.");
  }
};
REGISTER_MEDIAPIPE_GRAPH(FaceLandmarksSmoothing);
#endif

}  // namespace mediapipe

// the program entrance point, the main().
// If you have main() already, don't include this.
// Just reference RunMPPGraph() for the usage of 'graph'.
#include "../demo_run_graph_main.cc"

//
// IMPORTANT: The REGISTER_INPUT_STREAM_HANDLER() and REGISTER_CALCULATOR() problems...
// https://stackoverflow.com/questions/5202142/static-variable-initialization-over-a-library
//
// To make all registery static variables be instanced,
// You must set Linker command line Options: '/WHOLEARCHIVE:mediapipe.lib'
// https://docs.microsoft.com/en-us/cpp/build/reference/wholearchive-include-all-library-object-files?redirectedfrom=MSDN&view=msvc-160
//
// and because now the mediapipe.lib is a monster, you may like to enable 64-bit MSVC toolset
// https://docs.microsoft.com/en-us/cpp/build/how-to-enable-a-64-bit-visual-cpp-toolset-on-the-command-line?view=msvc-160
// C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat
//
// Or, to specfic using x64 MSVC toolset, open your .vcxproj file, find this line...
//  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
// then, insert this xml property...
//  <PropertyGroup>
//    <PreferredToolArchitecture>x64</PreferredToolArchitecture>
//  </PropertyGroup>
//
//
// If your tool don't suppor this function, move out all calculators out ot mediapile library,
// and add required calculators to your final executable project.
//