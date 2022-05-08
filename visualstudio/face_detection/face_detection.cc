#include "../calculator_graph_util.h"

// the program entrance point, the main().
// If you have main() already, don't include this.
// Just reference RunMPPGraph() for the usage of 'graph'.
#include "../demo_run_graph_main.cc"

// define graph loading function
DEFINE_LOAD_GRAPH("../../mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt")

namespace mediapipe {
DEFINE_SUBGRAPH(FaceDetectionShortRangeCpu, "face_detection_short_range_cpu.pbtxt");  // ../../mediapipe/modules/face_detection
  DEFINE_SUBGRAPH(FaceDetectionShortRangeCommon, "../../mediapipe/modules/face_detection/face_detection_short_range_common.pbtxt");
}