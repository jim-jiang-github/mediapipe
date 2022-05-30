#include "../calculator_graph_util.h"

// the program entrance point, the main().
// If you have main() already, don't include this.
// Just reference RunMPPGraph() for the usage of 'graph'.
#include "../demo_run_graph_main.cc"

// define graph loading function
DEFINE_LOAD_GRAPH("./objectron_desktop_live.pbtxt")

namespace mediapipe {

DEFINE_SUBGRAPH(ObjectronCpuSubgraph, "../../mediapipe/modules/objectron/objectron_cpu.pbtxt");
  DEFINE_SUBGRAPH(ObjectDetectionOidV4Subgraph, "./object_detection_oid_v4_cpu.pbtxt");
  DEFINE_SUBGRAPH(BoxLandmarkSubgraph, "../../mediapipe/modules/objectron/box_landmark_cpu.pbtxt");

DEFINE_SUBGRAPH(RendererSubgraph, "../../mediapipe/graphs/object_detection_3d/subgraphs/renderer_cpu.pbtxt");

}  // namespace mediapipe
