// Copyright 2022 andre.hl.chen@gmail.com
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "../calculator_graph_util.h"

// the program entrance point, the main().
// If you have main() already, don't include this.
// Just reference RunMPPGraph() for the usage of 'graph'.
#include "../demo_run_graph_main.cc"

// define graph loading function
DEFINE_LOAD_GRAPH("./face_effect_desktop_live.pbtxt")
//DEFINE_LOAD_GRAPH("./face_effect_desktop_live_vs_face_geometry.pbtxt") // test with face_geometry module

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

}  // namespace mediapipe
