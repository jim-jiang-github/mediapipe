#include "../calculator_graph_util.h"

// the program entrance point, the main().
// If you have main() already, don't include this.
// Just reference RunMPPGraph() for the usage of 'graph'.
#include "../demo_run_graph_main.cc"

// define graph loading function
DEFINE_LOAD_GRAPH("../../mediapipe/graphs/tracking/object_detection_tracking_desktop_live.pbtxt")

// subgraphs
namespace mediapipe {
DEFINE_SUBGRAPH(ObjectDetectionSubgraphCpu, "object_detection_cpu.pbtxt");

DEFINE_SUBGRAPH(ObjectTrackingSubgraphCpu, "../../mediapipe/graphs/tracking/subgraphs/object_tracking_cpu.pbtxt");
  DEFINE_SUBGRAPH(BoxTrackingSubgraphCpu, "../../mediapipe/graphs/tracking/subgraphs/box_tracking_cpu.pbtxt");

DEFINE_SUBGRAPH(RendererSubgraphCpu, "../../mediapipe/graphs/tracking/subgraphs/renderer_cpu.pbtxt");
}