#include "../calculator_graph_util.h"
DEFINE_LOAD_GRAPH("../../mediapipe/graphs/iris_tracking/iris_tracking_cpu.pbtxt")

namespace mediapipe {
DEFINE_SUBGRAPH(FaceLandmarkFrontCpu, "../../mediapipe/modules/face_landmark/face_landmark_front_cpu.pbtxt");
  DEFINE_SUBGRAPH(FaceDetectionShortRangeCpu, "../face_detection/face_detection_short_range_cpu.pbtxt");
    DEFINE_SUBGRAPH(FaceDetectionShortRangeCommon, "../../mediapipe/modules/face_detection/face_detection_short_range_common.pbtxt");
  DEFINE_SUBGRAPH(FaceDetectionFrontDetectionToRoi, "../../mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt");
  DEFINE_SUBGRAPH(FaceLandmarkCpu, "../../mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt");
    DEFINE_SUBGRAPH(FaceLandmarksModelLoader, "../face_mesh/face_landmarks_model_loader.pbtxt");
    DEFINE_SUBGRAPH(TensorsToFaceLandmarks, "../../mediapipe/modules/face_landmark/tensors_to_face_landmarks.pbtxt");
    DEFINE_SUBGRAPH(TensorsToFaceLandmarksWithAttention, "../../mediapipe/modules/face_landmark/tensors_to_face_landmarks_with_attention.pbtxt");
  DEFINE_SUBGRAPH(FaceLandmarkLandmarksToRoi, "../../mediapipe/modules/face_landmark/face_landmark_landmarks_to_roi.pbtxt");

DEFINE_SUBGRAPH(IrisLandmarkLeftAndRightCpu, "../../mediapipe/modules/iris_landmark/iris_landmark_left_and_right_cpu.pbtxt");
  DEFINE_SUBGRAPH(IrisLandmarkLandmarksToRoi, "../../mediapipe/modules/iris_landmark/iris_landmark_landmarks_to_roi.pbtxt");
  DEFINE_SUBGRAPH(IrisLandmarkCpu, "iris_landmark_cpu.pbtxt");  // ../../mediapipe/modules/iris_landmark

DEFINE_SUBGRAPH(IrisRendererCpu, "../../mediapipe/graphs/iris_tracking/subgraphs/iris_renderer_cpu.pbtxt");
}