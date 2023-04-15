// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <cstdlib>
#include "calculator_graph_util.h"
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
#include "mediapipe/framework/deps/status_macros.h"
#include <memory>
#include "HandGestureRecognition.h"

// subgraphs
namespace mediapipe {
    DEFINE_SUBGRAPH(HandLandmarkTrackingCpu, "C:/Users/Jim.Jiang/Documents/mediapipe/mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.pbtxt");
    DEFINE_SUBGRAPH(PalmDetectionCpu, "C:/Users/Jim.Jiang/Documents/mediapipe/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt");
    DEFINE_SUBGRAPH(PalmDetectionModelLoader, "C:/Users/Jim.Jiang/Documents/mediapipe/mediapipe/modules/palm_detection/palm_detection_model_loader.pbtxt");
    DEFINE_SUBGRAPH(PalmDetectionDetectionToRoi, "C:/Users/Jim.Jiang/Documents/mediapipe/mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt");
    DEFINE_SUBGRAPH(HandLandmarkCpu, "C:/Users/Jim.Jiang/Documents/mediapipe/mediapipe/modules/hand_landmark/hand_landmark_cpu.pbtxt");
    DEFINE_SUBGRAPH(HandLandmarkModelLoader, "C:/Users/Jim.Jiang/Documents/mediapipe/mediapipe/modules/hand_landmark/hand_landmark_model_loader.pbtxt");
    DEFINE_SUBGRAPH(HandLandmarkLandmarksToRoi, "C:/Users/Jim.Jiang/Documents/mediapipe/mediapipe/modules/hand_landmark/hand_landmark_landmarks_to_roi.pbtxt");
    DEFINE_SUBGRAPH(HandRendererSubgraph, "C:/Users/Jim.Jiang/Documents/mediapipe/mediapipe/graphs/hand_tracking/subgraphs/hand_renderer_cpu.pbtxt");
}
// resource root to locate tflite and other files
// see also mediapipe/mediapipe/util/resource_util_default.cc
constexpr char const* resource_root = "C:/Users/Jim.Jiang/Documents/mediapipe";

// name of file containing text format CalculatorGraphConfig proto
constexpr char const* calculator_graph_config_file = "C:/Users/Jim.Jiang/Documents/mediapipe/mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt";

ABSL_FLAG(std::string, input_video_path, "",
    "Full path of video to load. "
    "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
    "Full path of where to save result (.mp4 only). "
    "If not provided, show result in a window.");
mediapipe::CalculatorGraph graph;
std::unique_ptr<mediapipe::OutputStreamPoller> poller;
std::unique_ptr<mediapipe::OutputStreamPoller> poller_landmarks;

HandGestureRecognition mHandGestureRecognition;
auto const start_time = std::chrono::system_clock::now();
bool OneFrame(cv::Mat mat)
{
    // timestamp
    mediapipe::Timestamp const timestamp(get_elapsed_time_microseconds(start_time));

    // Wrap Mat into an ImageFrame.
    auto input_frame = abslx::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, mat.cols, mat.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    mat.copyTo(input_frame_mat);

    // Send image packet into the graph.
    graph.AddPacketToInputStream(
        "input_video", mediapipe::Adopt(input_frame.release()).At(timestamp));

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (poller->Next(&packet)) {
        auto& output_frame = packet.Get<mediapipe::ImageFrame>();

        // Convert back to opencv for display or saving.
        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

        //Gesture result = mHandGestureRecognition.GestureRecognition(singleHandGestureInfo);
        cv::imshow("MediaPipe", output_frame_mat);
        cv::waitKeyEx(1);
        return true;
    }
    else {
        return false;
    }
}

bool initHand()
{
    char resource_root_dir[128];
    sprintf(resource_root_dir, "--resource_root_dir=%s", resource_root);
    char* cstr = const_cast<char*>("");

    std::vector<char*> argvv;
    argvv.push_back(cstr);
    argvv.push_back(resource_root_dir);
    abslx::ParseCommandLine((int)argvv.size(), argvv.data());

    std::string calculator_graph_config_contents;
    mediapipe::file::GetContents(calculator_graph_config_file, &calculator_graph_config_contents);

    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);

    graph.Initialize(config);

    auto videoOutputStream = graph.AddOutputStreamPoller("output_video");
    poller = std::make_unique<mediapipe::OutputStreamPoller>(std::move(videoOutputStream.value()));
    auto videoOutputStream1 = graph.AddOutputStreamPoller("landmarks");
    poller_landmarks = std::make_unique<mediapipe::OutputStreamPoller>(std::move(videoOutputStream1.value()));
    graph.StartRun({});
    return true;
}
int main()
{
    initHand();

    cv::VideoCapture capture;
    capture.open(0);

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
    return 1;
}