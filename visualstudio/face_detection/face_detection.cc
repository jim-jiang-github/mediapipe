#include "../calculator_graph_util.h"
DEFINE_LOAD_GRAPH("../../mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt")

namespace mediapipe {
DEFINE_SUBGRAPH(FaceDetectionShortRangeCpu, "face_detection_short_range_cpu.pbtxt");  // ../../mediapipe/modules/face_detection
  DEFINE_SUBGRAPH(FaceDetectionShortRangeCommon, "../../mediapipe/modules/face_detection/face_detection_short_range_common.pbtxt");
}