#include "../calculator_graph_util.h"

// define graph loading function
DEFINE_LOAD_GRAPH("../../mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt");

// subgraphs
namespace mediapipe {
DEFINE_SUBGRAPH(HandLandmarkTrackingCpu, "../../mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.pbtxt");
  DEFINE_SUBGRAPH(PalmDetectionCpu, "../../mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt");
    DEFINE_SUBGRAPH(PalmDetectionModelLoader, "palm_detection_model_loader.pbtxt"); //../../mediapipe/modules/palm_detection 
  DEFINE_SUBGRAPH(PalmDetectionDetectionToRoi, "../../mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt");
  DEFINE_SUBGRAPH(HandLandmarkCpu, "hand_landmark_cpu.pbtxt"); // ../../mediapipe/modules/hand_landmark
    DEFINE_SUBGRAPH(HandLandmarkModelLoader, "hand_landmark_model_loader.pbtxt"); // ../../mediapipe/modules/hand_landmark
  DEFINE_SUBGRAPH(HandLandmarkLandmarksToRoi, "../../mediapipe/modules/hand_landmark/hand_landmark_landmarks_to_roi.pbtxt");

DEFINE_SUBGRAPH(HandRendererSubgraph, "../../mediapipe/graphs/hand_tracking/subgraphs/hand_renderer_cpu.pbtxt");
} // namespace mediapipe

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