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
//#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "hand_tracking/HandGestureRecognition.h"
constexpr char const* kWindowName = "MediaPipe";

constexpr char const* kInputStream = "input_video";
constexpr char const* kUserInput = "user_input";
constexpr char const* kOutputStream = "output_video";
constexpr char const* kOutputLandmarks = "landmarks";

ABSL_FLAG(std::string, input_video_path, "",
    "Full path of video to load. "
    "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
    "Full path of where to save result (.mp4 only). "
    "If not provided, show result in a window.");

absl::Status RunMPPGraph() {
    LOG(INFO) << "Initialize the calculator graph.";
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(init_calculator_graph(graph));

    LOG(INFO) << "Initialize the camera or load the video.";
    cv::VideoCapture capture;
    capture.open(0);
    RET_CHECK(capture.isOpened());

    cv::VideoWriter writer;
    const bool save_video = !absl::GetFlag(FLAGS_output_video_path).empty();
    if (!save_video) {
        cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if ((CV_MAJOR_VERSION*10+CV_MINOR_VERSION) >= 32)
        capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        capture.set(cv::CAP_PROP_FPS, 30);
#endif
    }

    LOG(INFO) << "Start running the calculator graph.";
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
        graph.AddOutputStreamPoller(kOutputStream));
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmarks,
        graph.AddOutputStreamPoller(kOutputLandmarks));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    LOG(INFO) << "Start grabbing and processing frames.";

    UserInput user_input;
    HandGestureRecognition handGestureRecognition;

    for (auto const start_time = std::chrono::system_clock::now();
        user_input.wait_key != 27; user_input.wait_key = cv::waitKeyEx(1)) {
        // close window after wait key
        if (cv::getWindowProperty(kWindowName, cv::WND_PROP_VISIBLE) < 1.0) {
            break;
        }

        // Capture opencv camera or video frame.
        cv::Mat camera_frame_raw;
        capture >> camera_frame_raw;
        if (!camera_frame_raw.empty()) {
            ++user_input.frame_id;
        }
        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

        cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);

        // timestamp
        mediapipe::Timestamp const timestamp(get_elapsed_time_microseconds(start_time));

        // Wrap Mat into an ImageFrame.
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        // Send image packet into the graph.
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release()).At(timestamp)));



        mediapipe::Packet packet;
        mediapipe::Packet packet_landmarks;
        if (!poller.Next(&packet))
            return absl::OkStatus();

        if (poller_landmarks.QueueSize() > 0)
        {
            if (poller_landmarks.Next(&packet_landmarks))
            {
                std::vector<mediapipe::NormalizedLandmarkList> output_landmarks = packet_landmarks.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
                int* hand_gesture_recognition_result = new int[output_landmarks.size()];
                std::vector<PoseInfo> hand_landmarks;
                hand_landmarks.clear();

                for (int m = 0; m < output_landmarks.size(); ++m)
                {
                    mediapipe::NormalizedLandmarkList single_hand_NormalizedLandmarkList = output_landmarks[m];

                    std::vector<PoseInfo> singleHandGestureInfo;
                    singleHandGestureInfo.clear();

                    for (int i = 0; i < single_hand_NormalizedLandmarkList.landmark_size(); ++i)
                    {
                        PoseInfo info;
                        const mediapipe::NormalizedLandmark landmark = single_hand_NormalizedLandmarkList.landmark(i);
                        info.x = landmark.x() * camera_frame.cols;
                        info.y = landmark.y() * camera_frame.rows;
                        singleHandGestureInfo.push_back(info);
                        hand_landmarks.push_back(info);
                    }

                    Gesture result = handGestureRecognition.GestureRecognition(singleHandGestureInfo);



                }
            }
        }


        //// Get the graph result packet, or stop if that fails.
        //mediapipe::Packet packet;
        //if (poller.Next(&packet)) {
        //    auto& output_frame = packet.Get<mediapipe::ImageFrame>();

        //    // Convert back to opencv for display or saving.
        //    cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        //    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

        //    cv::imshow(kWindowName, output_frame_mat);
        //}
        //else {
        //    break;
        //}
    }

    LOG(INFO) << "Shutting down...";
    cv::destroyAllWindows();
    if (writer.isOpened()) writer.release();
    MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());
    return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
    //google::InitGoogleLogging(argv[0]);

    // workaround tflite model path
    if (resource_root) {
        std::vector<char*> argvv(argc);
        bool find_resource_root = false;
        char resource_root_dir[128];
        sprintf(resource_root_dir, "--resource_root_dir=%s", resource_root);

        for (int i = 0; i < argc; ++i) {
            char* c = argvv[i] = argv[i];
            if (!find_resource_root) {
                /*     if ('-'==*c && '-'==*++c) {
                       ++c;
                     }*/

                if (0 == memcmp(c, "resource_root_dir=", 18)) {
                    /*   std::cout << "overwrite resource root: \"" << c+18 << "\" to \"" << resource_root << "\"\n";
                       argvv[i] = resource_root_dir;
                       find_resource_root = true;*/
                }
            }
        }

        if (!find_resource_root) {
            argvv.push_back(resource_root_dir);
        }
        absl::ParseCommandLine((int)argvv.size(), argvv.data());
    }
    else {
        absl::ParseCommandLine(argc, argv);
    }

    absl::Status run_status = RunMPPGraph();
    if (!run_status.ok()) {
        std::cout << "Failed to run the graph: " << run_status.message() << "\n";
        system("pause");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

void initHand()
{

}
