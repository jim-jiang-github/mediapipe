#include "../calculator_graph_util.h"

// resource root to locate tflite and other files
// see also mediapipe/mediapipe/util/resource_util_default.cc
constexpr char const* resource_root = "../";

// name of file containing text format CalculatorGraphConfig proto
constexpr char const* calculator_graph_config_file = "../../mediapipe/graphs/object_detection/object_detection_desktop_live.pbtxt";
abslx::Status init_calculator_graph(mediapipe::CalculatorGraph& graph) {
  mediapipe::CalculatorGraphConfig config;
  if (read_config_from_pbtxt(config, calculator_graph_config_file)) {
     download_mediapipe_asset_from_GCS("../mediapipe/models/ssdlite_object_detection.tflite");
     download_mediapipe_asset_from_GCS("../mediapipe/models/ssdlite_object_detection_labelmap.txt");
    return graph.Initialize(config);
  }
  return abslx::NotFoundError(calculator_graph_config_file);
}

// the program entrance point, the main().
// If you have main() already, don't include this.
// Just reference RunMPPGraph() for the usage of 'graph'.
#include "../demo_run_graph_main.cc"