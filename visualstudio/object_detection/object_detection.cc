#include "../calculator_graph_util.h"

// name of file containing text format CalculatorGraphConfig proto
constexpr char const* calculator_graph_config_file = "./object_detection_desktop_live.pbtxt";
absl::Status init_calculator_graph(mediapipe::CalculatorGraph& graph) {
  mediapipe::CalculatorGraphConfig config;
  if (read_config_from_pbtxt(config, calculator_graph_config_file)) {
    return graph.Initialize(config);
  }
  return absl::NotFoundError(calculator_graph_config_file);
}

// the program entrance point, the main().
// If you have main() already, don't include this.
// Just reference RunMPPGraph() for the usage of 'graph'.
#include "../demo_run_graph_main.cc"