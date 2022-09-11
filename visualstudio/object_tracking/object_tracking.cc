#include "../calculator_graph_util.h"

// resource root to locate tflite and other files
// see also mediapipe/mediapipe/util/resource_util_default.cc
constexpr char const* resource_root = "../";

// name of file containing text format CalculatorGraphConfig proto
constexpr char const* calculator_graph_config_file = "../../mediapipe/graphs/tracking/object_detection_tracking_desktop_live.pbtxt";

// subgraphs
namespace mediapipe {
DEFINE_SUBGRAPH(ObjectDetectionSubgraphCpu, "../../mediapipe/graphs/tracking/subgraphs/object_detection_cpu.pbtxt");
DEFINE_SUBGRAPH(ObjectTrackingSubgraphCpu, "../../mediapipe/graphs/tracking/subgraphs/object_tracking_cpu.pbtxt");
  DEFINE_SUBGRAPH(BoxTrackingSubgraphCpu, "../../mediapipe/graphs/tracking/subgraphs/box_tracking_cpu.pbtxt");
DEFINE_SUBGRAPH(RendererSubgraphCpu, "../../mediapipe/graphs/tracking/subgraphs/renderer_cpu.pbtxt");
}

absl::Status init_calculator_graph(mediapipe::CalculatorGraph& graph) {
  mediapipe::CalculatorGraphConfig config;
  if (read_config_from_pbtxt(config, calculator_graph_config_file)) {
    download_mediapipe_asset_from_GCS("../mediapipe/models/ssdlite_object_detection.tflite");
    download_mediapipe_asset_from_GCS("../mediapipe/models/ssdlite_object_detection_labelmap.txt");
    return graph.Initialize(config);
  }
  return absl::NotFoundError(calculator_graph_config_file);
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