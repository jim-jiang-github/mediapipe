#include "../calculator_graph_util.h"
DEFINE_LOAD_GRAPH("../../mediapipe/graphs/holistic_tracking/holistic_tracking_cpu.pbtxt")

// subgraphs
namespace mediapipe {
DEFINE_SUBGRAPH(HolisticLandmarkCpu, "../../mediapipe/modules/holistic_landmark/holistic_landmark_cpu.pbtxt");
  DEFINE_SUBGRAPH(PoseLandmarkCpu, "../../mediapipe/modules/pose_landmark/pose_landmark_cpu.pbtxt");
    DEFINE_SUBGRAPH(PoseDetectionCpu, "../pose_tracking/pose_detection_cpu.pbtxt"); // ../../mediapipe/modules/pose_detection
    DEFINE_SUBGRAPH(PoseDetectionToRoi, "../../mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt");
    DEFINE_SUBGRAPH(PoseLandmarkByRoiCpu, "../../mediapipe/modules/pose_landmark/pose_landmark_by_roi_cpu.pbtxt");
      DEFINE_SUBGRAPH(PoseLandmarkModelLoader, "../pose_tracking/pose_landmark_model_loader.pbtxt"); // ../../mediapipe/modules/pose_landmark
      DEFINE_SUBGRAPH(TensorsToPoseLandmarksAndSegmentation, "../../mediapipe/modules/pose_landmark/tensors_to_pose_landmarks_and_segmentation.pbtxt");
      DEFINE_SUBGRAPH(PoseLandmarksAndSegmentationInverseProjection, "../../mediapipe/modules/pose_landmark/pose_landmarks_and_segmentation_inverse_projection.pbtxt");
    DEFINE_SUBGRAPH(PoseLandmarkFiltering, "../../mediapipe/modules/pose_landmark/pose_landmark_filtering.pbtxt");
    DEFINE_SUBGRAPH(PoseLandmarksToRoi, "../../mediapipe/modules/pose_landmark/pose_landmarks_to_roi.pbtxt");
    DEFINE_SUBGRAPH(PoseSegmentationFiltering, "../../mediapipe/modules/pose_landmark/pose_segmentation_filtering.pbtxt");

  DEFINE_SUBGRAPH(HandLandmarksLeftAndRightCpu, "../../mediapipe/modules/holistic_landmark/hand_landmarks_left_and_right_cpu.pbtxt"); 
   DEFINE_SUBGRAPH(HandLandmarksFromPoseCpu, "../../mediapipe/modules/holistic_landmark/hand_landmarks_from_pose_cpu.pbtxt");
     DEFINE_SUBGRAPH(HandVisibilityFromHandLandmarksFromPose, "../../mediapipe/modules/holistic_landmark/hand_visibility_from_hand_landmarks_from_pose.pbtxt");
     DEFINE_SUBGRAPH(HandLandmarksFromPoseToRecropRoi, "../../mediapipe/modules/holistic_landmark/hand_landmarks_from_pose_to_recrop_roi.pbtxt");
     DEFINE_SUBGRAPH(HandRecropByRoiCpu, "hand_recrop_by_roi_cpu.pbtxt"); // ../../mediapipe/modules/holistic_landmark
     DEFINE_SUBGRAPH(HandTracking, "../../mediapipe/modules/holistic_landmark/hand_tracking.pbtxt");
       DEFINE_SUBGRAPH(HandLandmarksToRoi, "../../mediapipe/modules/holistic_landmark/hand_landmarks_to_roi.pbtxt");
     DEFINE_SUBGRAPH(HandLandmarkCpu, "../hand_tracking/hand_landmark_cpu.pbtxt"); // ../../mediapipe/modules/hand_landmark
       DEFINE_SUBGRAPH(HandLandmarkModelLoader, "../hand_tracking/hand_landmark_model_loader.pbtxt"); // ../../mediapipe/modules/hand_landmark

  DEFINE_SUBGRAPH(FaceLandmarksFromPoseCpu, "../../mediapipe/modules/holistic_landmark/face_landmarks_from_pose_cpu.pbtxt");
    DEFINE_SUBGRAPH(FaceLandmarksFromPoseToRecropRoi, "../../mediapipe/modules/holistic_landmark/face_landmarks_from_pose_to_recrop_roi.pbtxt");
    DEFINE_SUBGRAPH(FaceDetectionShortRangeByRoiCpu, "face_detection_short_range_by_roi_cpu.pbtxt"); // ../../mediapipe/modules/face_detection
      DEFINE_SUBGRAPH(FaceDetectionShortRangeCommon, "../../mediapipe/modules/face_detection/face_detection_short_range_common.pbtxt");
    DEFINE_SUBGRAPH(FaceDetectionFrontDetectionsToRoi, "../../mediapipe/modules/holistic_landmark/face_detection_front_detections_to_roi.pbtxt");
    DEFINE_SUBGRAPH(FaceTracking, "../../mediapipe/modules/holistic_landmark/face_tracking.pbtxt");
      DEFINE_SUBGRAPH(FaceLandmarksToRoi, "../../mediapipe/modules/holistic_landmark/face_landmarks_to_roi.pbtxt");
    DEFINE_SUBGRAPH(FaceLandmarkCpu, "../../mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt");
      DEFINE_SUBGRAPH(FaceLandmarksModelLoader, "../face_mesh/face_landmarks_model_loader.pbtxt"); // ../../mediapipe/modules/face_landmark
      DEFINE_SUBGRAPH(TensorsToFaceLandmarks, "../../mediapipe/modules/face_landmark/tensors_to_face_landmarks.pbtxt");
      DEFINE_SUBGRAPH(TensorsToFaceLandmarksWithAttention, "../../mediapipe/modules/face_landmark/tensors_to_face_landmarks_with_attention.pbtxt");

DEFINE_SUBGRAPH(HolisticTrackingToRenderData, "../../mediapipe/graphs/holistic_tracking/holistic_tracking_to_render_data.pbtxt");
  DEFINE_SUBGRAPH(HandWristForPose, "../../mediapipe/modules/holistic_landmark/hand_wrist_for_pose.pbtxt");
} // namespace mediapipe
