//
// taken from ./mediapipe/examples/desktop/demo_run_graph_main.cc
// program entry point.
//

// Copyright 2019 The MediaPipe Authors.
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
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "hand_tracking/HandGestureRecognition.h"


abslx::Status RunMPPGraph() {


    HandGestureRecognition handGestureRecognition;
    auto const start_time = std::chrono::system_clock::now();

    cv::VideoCapture capture;
    capture.open(0);
    RET_CHECK(capture.isOpened());


    LOG(INFO) << "Start grabbing and processing frames.";

    while (true) {

        // Capture opencv camera or video frame.
        cv::Mat camera_frame_raw;
        capture >> camera_frame_raw;
        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

        cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
        OneFrame(camera_frame);
    }

    LOG(INFO) << "Shutting down...";
    cv::destroyAllWindows();
    MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());
    return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
    if (initHand()) {
        abslx::Status run_status = RunMPPGraph();
        if (!run_status.ok()) {
            std::cout << "11Failed to run the graph: " << run_status.message() << "\n";
            system("pause");
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

bool initHand();

bool OneFrame(cv::Mat mat);
