#include "../calculator_graph_util.h"

// the program entrance point, the main().
// If you have main() already, don't include this.
// Just reference RunMPPGraph() for the usage of 'graph'.
#include "../demo_run_graph_main.cc"

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
