#include "../calculator_graph_util.h"

// the program entrance point, the main().
// If you have main() already, don't include this.
// Just reference RunMPPGraph() for the usage of 'graph'.
#include "../demo_run_graph_main.cc"

// define graph loading function
DEFINE_LOAD_GRAPH("../../mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt")

// subgraphs
namespace mediapipe {
DEFINE_SUBGRAPH(PoseLandmarkCpu, "../../mediapipe/modules/pose_landmark/pose_landmark_cpu.pbtxt");
  DEFINE_SUBGRAPH(PoseDetectionCpu, "pose_detection_cpu.pbtxt"); // ../../mediapipe/modules/pose_detection
  DEFINE_SUBGRAPH(PoseDetectionToRoi, "../../mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt");
  DEFINE_SUBGRAPH(PoseLandmarkByRoiCpu, "../../mediapipe/modules/pose_landmark/pose_landmark_by_roi_cpu.pbtxt");
    DEFINE_SUBGRAPH(PoseLandmarkModelLoader, "pose_landmark_model_loader.pbtxt"); // ../../mediapipe/modules/pose_landmark
    DEFINE_SUBGRAPH(TensorsToPoseLandmarksAndSegmentation, "../../mediapipe/modules/pose_landmark/tensors_to_pose_landmarks_and_segmentation.pbtxt");
    DEFINE_SUBGRAPH(PoseLandmarksAndSegmentationInverseProjection, "../../mediapipe/modules/pose_landmark/pose_landmarks_and_segmentation_inverse_projection.pbtxt");
  DEFINE_SUBGRAPH(PoseLandmarkFiltering, "../../mediapipe/modules/pose_landmark/pose_landmark_filtering.pbtxt");
  DEFINE_SUBGRAPH(PoseLandmarksToRoi, "../../mediapipe/modules/pose_landmark/pose_landmarks_to_roi.pbtxt");
  DEFINE_SUBGRAPH(PoseSegmentationFiltering, "../../mediapipe/modules/pose_landmark/pose_segmentation_filtering.pbtxt");

DEFINE_SUBGRAPH(PoseRendererCpu, "../../mediapipe/graphs/pose_tracking/subgraphs/pose_renderer_cpu.pbtxt");
} // namespace mediapipe
