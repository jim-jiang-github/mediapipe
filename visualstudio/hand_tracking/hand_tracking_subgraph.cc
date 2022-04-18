#include "../calculator_graph_util.h"
DEFINE_LOAD_GRAPH("hand_tracking_desktop_live.pbtxt");

// subgraphs
namespace mediapipe {
DEFINE_SUBGRAPH(HandLandmarkTrackingCpu, "hand_landmark_tracking_cpu.pbtxt");
  DEFINE_SUBGRAPH(PalmDetectionCpu, "palm_detection_cpu.pbtxt");
    DEFINE_SUBGRAPH(PalmDetectionModelLoader, "palm_detection_model_loader.pbtxt");
  DEFINE_SUBGRAPH(PalmDetectionDetectionToRoi, "palm_detection_detection_to_roi.pbtxt");
  DEFINE_SUBGRAPH(HandLandmarkCpu, "hand_landmark_cpu.pbtxt");
    DEFINE_SUBGRAPH(HandLandmarkModelLoader, "hand_landmark_model_loader.pbtxt");
  DEFINE_SUBGRAPH(HandLandmarkLandmarksToRoi, "hand_landmark_landmarks_to_roi.pbtxt");

DEFINE_SUBGRAPH(HandRendererSubgraph, "hand_renderer_cpu.pbtxt");
} // namespace mediapipe
