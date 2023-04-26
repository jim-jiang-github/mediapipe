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
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/deps/status_macros.h"
#include <memory>
#include "HandGestureRecognition.h"

// subgraphs
namespace mediapipe {
    DEFINE_SUBGRAPH(HandLandmarkTrackingCpu, "C:/mediapipe/mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.pbtxt");
    DEFINE_SUBGRAPH(PalmDetectionCpu, "C:/mediapipe/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt");
    DEFINE_SUBGRAPH(PalmDetectionModelLoader, "C:/mediapipe/mediapipe/modules/palm_detection/palm_detection_model_loader.pbtxt");
    DEFINE_SUBGRAPH(PalmDetectionDetectionToRoi, "C:/mediapipe/mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt");
    DEFINE_SUBGRAPH(HandLandmarkCpu, "C:/mediapipe/mediapipe/modules/hand_landmark/hand_landmark_cpu.pbtxt");
    DEFINE_SUBGRAPH(HandLandmarkModelLoader, "C:/mediapipe/mediapipe/modules/hand_landmark/hand_landmark_model_loader.pbtxt");
    DEFINE_SUBGRAPH(HandLandmarkLandmarksToRoi, "C:/mediapipe/mediapipe/modules/hand_landmark/hand_landmark_landmarks_to_roi.pbtxt");
    DEFINE_SUBGRAPH(HandRendererSubgraph, "C:/mediapipe/mediapipe/graphs/hand_tracking/subgraphs/hand_renderer_cpu.pbtxt");
}
// resource root to locate tflite and other files
// see also mediapipe/mediapipe/util/resource_util_default.cc
constexpr char const* resource_root = "C:/mediapipe";

// name of file containing text format CalculatorGraphConfig proto
constexpr char const* calculator_graph_config_file = "C:/mediapipe/mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt";

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
bool lastIsClick = false;
bool lastIsDown = false;
int lastIsClickCount = 0;
int lastIsDownCount = 0;
bool lastIsReset = false;
bool isFirstWheel = false;
float lastX, lastY, lastWY;
int width, height, appendX, appendY;
std::vector<PoseInfo> mDebounceCache;
std::vector<PoseInfo> mDebounceCache1;
int mDebounceThreshold = 10;
int mouseSpeed = 2;
int mDistanceThreshold = 10;
bool isDoubleFiltering = false;
float distance(PoseInfo p1, PoseInfo p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}
PoseInfo debounce(PoseInfo point) {
    PoseInfo res = point;
    if (mDebounceCache.size() > 0)
    {
        res = mDebounceCache.back();
    }
    mDebounceCache.push_back(point);
    if (mDebounceCache.size() > 5) {
        mDebounceCache.erase(mDebounceCache.begin());
        float sum_x = 0, sum_y = 0;
        for (int i = 0; i < mDebounceCache.size(); i++) {
            sum_x += mDebounceCache[i].x;
            sum_y += mDebounceCache[i].y;
        }
        res = PoseInfo{ sum_x / mDebounceCache.size(), sum_y / mDebounceCache.size() };
    }
    return res;
}
PoseInfo debounce1(PoseInfo point) {
    PoseInfo res = point;
    if (mDebounceCache1.size() > 0)
    {
        res = mDebounceCache1.back();
    }
    mDebounceCache1.push_back(point);
    if (mDebounceCache1.size() > 5) {
        mDebounceCache1.erase(mDebounceCache1.begin());
        float sum_x = 0, sum_y = 0;
        for (int i = 0; i < mDebounceCache1.size(); i++) {
            sum_x += mDebounceCache1[i].x;
            sum_y += mDebounceCache1[i].y;
        }
        res = PoseInfo{ sum_x / mDebounceCache1.size(), sum_y / mDebounceCache1.size() };
    }
    return res;
}
PoseInfo centerPoint(const std::vector<PoseInfo>& points) {
    std::vector<int> indexs{ 0, 1, 5, 9, 13, 17 };
    int n = indexs.size();
    PoseInfo sum{ 0, 0 };
    for (int i = 0; i < n; ++i) {
        int index = indexs[i];
        sum.x += points[i].x;
        sum.y += points[i].y;
    }
    return PoseInfo{ sum.x / n, sum.y / n };
}
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

    mediapipe::Packet packet;
    if (!poller->Next(&packet))
    {
        return false;
    }
    cv::Mat output_frame_mat = mediapipe::formats::MatView(&packet.Get<mediapipe::ImageFrame>());
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    cv::imshow("MediaPipe", output_frame_mat);
    cv::waitKeyEx(1);

    mediapipe::Packet packet_landmarks;
    if (poller_landmarks->QueueSize() <= 0 || !poller_landmarks->Next(&packet_landmarks))
    {
        return false;
    }
    std::vector<mediapipe::NormalizedLandmarkList> output_landmarks = packet_landmarks.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
    if (output_landmarks.size() != 1)
    {
        return false;
    }
    std::vector<PoseInfo> hand_landmarks;
    hand_landmarks.clear();
    mediapipe::NormalizedLandmarkList single_hand_NormalizedLandmarkList = output_landmarks[0];
    std::vector<PoseInfo> singleHandGestureInfo;
    singleHandGestureInfo.clear();
    auto s = single_hand_NormalizedLandmarkList.landmark_size();
    for (int i = 0; i < single_hand_NormalizedLandmarkList.landmark_size(); ++i)
    {
        PoseInfo info;
        const mediapipe::NormalizedLandmark landmark = single_hand_NormalizedLandmarkList.landmark(i);
        info.x = landmark.x() * output_frame_mat.cols;
        info.y = landmark.y() * output_frame_mat.rows;
        singleHandGestureInfo.push_back(info);
        hand_landmarks.push_back(info);
        auto gesture = mHandGestureRecognition.GestureRecognition(hand_landmarks);
        if (gesture == Gesture::One || gesture == Gesture::Two || gesture == Gesture::Five || gesture == Gesture::Three)
        {
            auto poseInfo = centerPoint(singleHandGestureInfo);
            if (isDoubleFiltering)
            {
                poseInfo = debounce1(poseInfo);
            }
            float x = (int)poseInfo.x * mouseSpeed * width / output_frame_mat.cols;
            float y = (int)poseInfo.y * mouseSpeed * height / output_frame_mat.rows;
            //std::cout << "x" << x << "y" << y << std::endl;
            if (gesture == Gesture::Five)
            {
                appendX = width / 2 - x;
                appendY = height / 2 - y;
            }
            x = x + appendX;
            y = y + appendY;
            poseInfo = PoseInfo{ x, y };
            if (gesture == Gesture::Five)
            {
                lastX = poseInfo.x;
                lastY = poseInfo.y;
                if (!lastIsReset)
                {
                    lastIsReset = true;
                    std::cout << "Reset" << std::endl;
                }
            }
            else
            {
                lastIsReset = false;
                /* auto d = distance(poseInfo, { lastX, lastY });
                 if (d > mDistanceThreshold)
                 {
                     lastX = poseInfo.x;
                     lastY = poseInfo.y;
                 }*/
                poseInfo = debounce(poseInfo);
                lastX = poseInfo.x;
                lastY = poseInfo.y;
                if (gesture == Gesture::Three)
                {
                    if (!isFirstWheel)
                    {
                        isFirstWheel = true;
                        lastWY = lastY;
                        keybd_event(VK_CONTROL, 0, 0, 0);
                    }
                    std::cout << "lastY" << lastY << " lastWY" << lastWY << " -" << lastY - lastWY << std::endl;
                    if (lastY - lastWY > 0)
                    {
                        mouse_event(MOUSEEVENTF_WHEEL, 0, 0, 120, 0);
                    }
                    else
                    {
                        mouse_event(MOUSEEVENTF_WHEEL, 0, 0, -120, 0);
                    }
                    break;
                }
            }
            keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0);
            isFirstWheel = false;
            SetCursorPos(lastX, lastY);

            lastIsClick = false;
            lastIsClickCount = 0;
            if (gesture == Gesture::Two)
            {
                if (!lastIsDown)
                {
                    lastIsDownCount++;
                    if (lastIsDownCount >= 3)
                    {
                        lastIsDownCount = 0;
                        lastIsDown = true;
                        mouse_event(MOUSEEVENTF_LEFTDOWN, lastX, lastY, 0, 0);
                        std::cout << "Down" << std::endl;
                    }
                }
            }
            else
            {
                if (lastIsDown) {
                    lastIsDownCount = 0;
                    lastIsDown = false;
                    mouse_event(MOUSEEVENTF_LEFTUP, lastX, lastY, 0, 0);
                    std::cout << "Up" << std::endl;
                }
            }
            //std::cout << "Move" << std::endl;
            break;
        };
        if (gesture == Gesture::Fist)
        {
            if (!lastIsClick)
            {
                lastIsClickCount++;
                if (lastIsClickCount >= 3)
                {
                    lastIsClickCount = 0;
                    lastIsClick = true;
                    mouse_event(MOUSEEVENTF_LEFTDOWN, lastX, lastY, 0, 0);
                    mouse_event(MOUSEEVENTF_LEFTUP, lastX, lastY, 0, 0);
                    std::cout << "Click" << std::endl;
                }
            }
        }
    }

    //mediapipe::Packet packet_handedness;
    //if (poller_handedness->QueueSize() <= 0 || !poller_handedness->Next(&packet_handedness))
    //{
    //    return false;
    //}
    //if (!packet_handedness.IsEmpty()) {
    //    auto status = packet_handedness.ValidateAsType<mediapipe::ClassificationList>();
    //    if (status.ok())
    //    {
    //        const auto& handedness = packet_handedness.Get<mediapipe::ClassificationList>();
    //    }
    //    // 处理 handedness 数据流的数据
    //}
    return true;
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
    HWND hWnd = GetDesktopWindow(); // 获取桌面窗口句柄
    HDC hDc = GetDC(hWnd); // 获取桌面窗口设备上下文句柄
    int dpi = GetDpiForWindow(hWnd); // 获取桌面窗口的 DPI 值
    ReleaseDC(hWnd, hDc);
    width = GetSystemMetrics(SM_CXSCREEN);
    height = GetSystemMetrics(SM_CYSCREEN);
    //width = GetSystemMetrics(SM_CXSCREEN) * dpi / 96;
    //height = GetSystemMetrics(SM_CYSCREEN) * dpi / 96;
    return true;
}
int main()
{
    int cam;
    std::string doubleFiltering;

    std::cout << "Select Cam (0~N): ";
    std::cin >> cam;
    std::cout << "Select mouse move speed (2~N): ";
    std::cin >> mouseSpeed;
    std::cout << "Enable double filtering? y/n:" << doubleFiltering << std::endl;
    std::cin >> doubleFiltering;
    isDoubleFiltering = doubleFiltering == "y" ? true : false;
    std::cout << "Cam index:" << cam << " Mouse speed:" << mouseSpeed << " Enable double filtering:" << (isDoubleFiltering ? "true" : "false") << std::endl;

    initHand();

    cv::VideoCapture capture;
    capture.open(cam);
    std::cout << "capture.open(" << cam << ")" << std::endl;

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