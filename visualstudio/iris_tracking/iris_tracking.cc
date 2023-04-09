#include "../calculator_graph_util.h"
#include "../register_options.h"

// resource root to locate tflite and other files
// see also mediapipe/mediapipe/util/resource_util_default.cc
constexpr char const* resource_root = "../";

// name of file containing text format CalculatorGraphConfig proto
constexpr char const* calculator_graph_config_file = "../../mediapipe/graphs/iris_tracking/iris_tracking_cpu.pbtxt";

namespace mediapipe {
DEFINE_SUBGRAPH(FaceLandmarkFrontCpu, "../../mediapipe/modules/face_landmark/face_landmark_front_cpu.pbtxt");
  DEFINE_SUBGRAPH(FaceDetectionShortRangeCpu, "../../mediapipe/modules/face_detection/face_detection_short_range_cpu.pbtxt");
    DEFINE_SUBGRAPH(FaceDetectionShortRange, "../../mediapipe/modules/face_detection/face_detection_short_range.pbtxt");
      DEFINE_SUBGRAPH(FaceDetection, "../../mediapipe/modules/face_detection/face_detection.pbtxt");

  DEFINE_SUBGRAPH(FaceDetectionFrontDetectionToRoi, "../../mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt");
  DEFINE_SUBGRAPH(FaceLandmarkCpu, "../../mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt");
    DEFINE_SUBGRAPH(FaceLandmarksModelLoader, "../../mediapipe/modules/face_landmark/face_landmarks_model_loader.pbtxt");
    DEFINE_SUBGRAPH(TensorsToFaceLandmarks, "../../mediapipe/modules/face_landmark/tensors_to_face_landmarks.pbtxt");
    DEFINE_SUBGRAPH(TensorsToFaceLandmarksWithAttention, "../../mediapipe/modules/face_landmark/tensors_to_face_landmarks_with_attention.pbtxt");
  DEFINE_SUBGRAPH(FaceLandmarkLandmarksToRoi, "../../mediapipe/modules/face_landmark/face_landmark_landmarks_to_roi.pbtxt");

DEFINE_SUBGRAPH(IrisLandmarkLeftAndRightCpu, "../../mediapipe/modules/iris_landmark/iris_landmark_left_and_right_cpu.pbtxt");
  DEFINE_SUBGRAPH(IrisLandmarkLandmarksToRoi, "../../mediapipe/modules/iris_landmark/iris_landmark_landmarks_to_roi.pbtxt");
  DEFINE_SUBGRAPH(IrisLandmarkCpu, "../../mediapipe/modules/iris_landmark/iris_landmark_cpu.pbtxt");

DEFINE_SUBGRAPH(IrisRendererCpu, "../../mediapipe/graphs/iris_tracking/subgraphs/iris_renderer_cpu.pbtxt");
}

abslx::Status init_calculator_graph(mediapipe::CalculatorGraph& graph) {
  // register options
  register_face_detection_options();

  // config
  mediapipe::CalculatorGraphConfig config;
  if (read_config_from_pbtxt(config, calculator_graph_config_file)) {
    download_mediapipe_asset_from_GCS("../mediapipe/modules/face_detection/face_detection_short_range.tflite");
    download_mediapipe_asset_from_GCS("../mediapipe/modules/face_landmark/face_landmark.tflite");
    download_mediapipe_asset_from_GCS("../mediapipe/modules/iris_landmark/iris_landmark.tflite");
    return graph.Initialize(config);
  }
  return abslx::NotFoundError(calculator_graph_config_file);
}

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