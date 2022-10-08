#include "../calculator_graph_util.h"
#include "../register_options.h"

#include "absl/flags/parse.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

//
// see discussion https://github.com/google/mediapipe/issues/1162#issuecomment-1123137462
//
// It's NOT pretty, but that's the way mediapipe subgraph system goes...
// to load a calculator graph, you must register all subgraphs, all calculators before hand.
// if you check REGISTER_MEDIAPIPE_GRAPH(), REGISTER_CALCULATOR() and REGISTER_INPUT_STREAM_HANDLER(),
// local static dangling pointers are created to trigger registery work.
//
// IMPORTANT: The REGISTER_INPUT_STREAM_HANDLER() and REGISTER_CALCULATOR() problems...
// https://stackoverflow.com/questions/5202142/static-variable-initialization-over-a-library
//
// To make all registery static variables be instanced,
// You must set Linker command line Options: '/WHOLEARCHIVE:mediapipe.lib'
// https://docs.microsoft.com/en-us/cpp/build/reference/wholearchive-include-all-library-object-files?redirectedfrom=MSDN&view=msvc-160
//
// and because now the mediapipe.lib is a monster, you may like to enable 64-bit MSVC toolset
// https://docs.microsoft.com/en-us/cpp/build/how-to-enable-a-64-bit-visual-cpp-toolset-on-the-command-line?view=msvc-160
// C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat
//
// Or, to specfic using x64 MSVC toolset, open your .vcxproj file, find this line...
//  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
// then, insert this xml property...
//  <PropertyGroup>
//    <PreferredToolArchitecture>x64</PreferredToolArchitecture>
//  </PropertyGroup>
//
//
// If your tool don't suppor this function, move out all calculators out ot mediapile library,
// and add required calculators to your final executable project.
//
namespace mediapipe {
// face detection
DEFINE_SUBGRAPH(FaceDetectionShortRangeCpu, "../../mediapipe/modules/face_detection/face_detection_short_range_cpu.pbtxt");
  DEFINE_SUBGRAPH(FaceDetectionShortRange, "../../mediapipe/modules/face_detection/face_detection_short_range.pbtxt");
    DEFINE_SUBGRAPH(FaceDetection, "../../mediapipe/modules/face_detection/face_detection.pbtxt");

// face mesh
DEFINE_SUBGRAPH(FaceLandmarkFrontCpu, "../../mediapipe/modules/face_landmark/face_landmark_front_cpu.pbtxt");
//  DEFINE_SUBGRAPH(FaceDetectionShortRangeCpu, "../../mediapipe/modules/face_detection/face_detection_short_range_cpu.pbtxt");
//    DEFINE_SUBGRAPH(FaceDetectionShortRange, "../../mediapipe/modules/face_detection/face_detection_short_range.pbtxt");
//      DEFINE_SUBGRAPH(FaceDetection, "../../mediapipe/modules/face_detection/face_detection.pbtxt");
  DEFINE_SUBGRAPH(FaceDetectionFrontDetectionToRoi, "../../mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt");
  DEFINE_SUBGRAPH(FaceLandmarkCpu, "../../mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt");
    DEFINE_SUBGRAPH(FaceLandmarksModelLoader, "../../mediapipe/modules/face_landmark/face_landmarks_model_loader.pbtxt");
    DEFINE_SUBGRAPH(TensorsToFaceLandmarks, "../../mediapipe/modules/face_landmark/tensors_to_face_landmarks.pbtxt");
    DEFINE_SUBGRAPH(TensorsToFaceLandmarksWithAttention, "../../mediapipe/modules/face_landmark/tensors_to_face_landmarks_with_attention.pbtxt");
  DEFINE_SUBGRAPH(FaceLandmarkLandmarksToRoi, "../../mediapipe/modules/face_landmark/face_landmark_landmarks_to_roi.pbtxt");
DEFINE_SUBGRAPH(FaceRendererCpu, "../../mediapipe/graphs/face_mesh/subgraphs/face_renderer_cpu.pbtxt");

// iris tracking
//DEFINE_SUBGRAPH(FaceLandmarkFrontCpu, "../../mediapipe/modules/face_landmark/face_landmark_front_cpu.pbtxt");
//  DEFINE_SUBGRAPH(FaceDetectionShortRangeCpu, "../../mediapipe/modules/face_detection/face_detection_short_range_cpu.pbtxt");
//    DEFINE_SUBGRAPH(FaceDetectionShortRange, "../../mediapipe/modules/face_detection/face_detection_short_range_cpu.pbtxt");
//      DEFINE_SUBGRAPH(FaceDetection, "../../mediapipe/modules/face_detection/face_detection.pbtxt");
//  DEFINE_SUBGRAPH(FaceDetectionFrontDetectionToRoi, "../../mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt");
//  DEFINE_SUBGRAPH(FaceLandmarkCpu, "../../mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt");
//    DEFINE_SUBGRAPH(FaceLandmarksModelLoader, "../../mediapipe/modules/face_landmark/face_landmarks_model_loader.pbtxt");
//    DEFINE_SUBGRAPH(TensorsToFaceLandmarks, "../../mediapipe/modules/face_landmark/tensors_to_face_landmarks.pbtxt");
//    DEFINE_SUBGRAPH(TensorsToFaceLandmarksWithAttention, "../../mediapipe/modules/face_landmark/tensors_to_face_landmarks_with_attention.pbtxt");
//  DEFINE_SUBGRAPH(FaceLandmarkLandmarksToRoi, "../../mediapipe/modules/face_landmark/face_landmark_landmarks_to_roi.pbtxt");
DEFINE_SUBGRAPH(IrisLandmarkLeftAndRightCpu, "../../mediapipe/modules/iris_landmark/iris_landmark_left_and_right_cpu.pbtxt");
  DEFINE_SUBGRAPH(IrisLandmarkLandmarksToRoi, "../../mediapipe/modules/iris_landmark/iris_landmark_landmarks_to_roi.pbtxt");
  DEFINE_SUBGRAPH(IrisLandmarkCpu, "../../mediapipe/modules/iris_landmark/iris_landmark_cpu.pbtxt");
DEFINE_SUBGRAPH(IrisRendererCpu, "../../mediapipe/graphs/iris_tracking/subgraphs/iris_renderer_cpu.pbtxt");

// hand tracking
DEFINE_SUBGRAPH(HandLandmarkTrackingCpu, "../../mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.pbtxt");
  DEFINE_SUBGRAPH(PalmDetectionCpu, "../../mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt");
    DEFINE_SUBGRAPH(PalmDetectionModelLoader, "../../mediapipe/modules/palm_detection/palm_detection_model_loader.pbtxt");
  DEFINE_SUBGRAPH(PalmDetectionDetectionToRoi, "../../mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt");
  DEFINE_SUBGRAPH(HandLandmarkCpu, "../../mediapipe/modules/hand_landmark/hand_landmark_cpu.pbtxt");
    DEFINE_SUBGRAPH(HandLandmarkModelLoader, "../../mediapipe/modules/hand_landmark/hand_landmark_model_loader.pbtxt");
  DEFINE_SUBGRAPH(HandLandmarkLandmarksToRoi, "../../mediapipe/modules/hand_landmark/hand_landmark_landmarks_to_roi.pbtxt");
DEFINE_SUBGRAPH(HandRendererSubgraph, "../../mediapipe/graphs/hand_tracking/subgraphs/hand_renderer_cpu.pbtxt");

// pose tracking
DEFINE_SUBGRAPH(PoseLandmarkCpu, "../../mediapipe/modules/pose_landmark/pose_landmark_cpu.pbtxt");
  DEFINE_SUBGRAPH(PoseDetectionCpu, "../../mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt");
  DEFINE_SUBGRAPH(PoseDetectionToRoi, "../../mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt");
  DEFINE_SUBGRAPH(PoseLandmarkByRoiCpu, "../../mediapipe/modules/pose_landmark/pose_landmark_by_roi_cpu.pbtxt");
    DEFINE_SUBGRAPH(PoseLandmarkModelLoader, "../../mediapipe/modules/pose_landmark/pose_landmark_model_loader.pbtxt");
    DEFINE_SUBGRAPH(TensorsToPoseLandmarksAndSegmentation, "../../mediapipe/modules/pose_landmark/tensors_to_pose_landmarks_and_segmentation.pbtxt");
    DEFINE_SUBGRAPH(PoseLandmarksAndSegmentationInverseProjection, "../../mediapipe/modules/pose_landmark/pose_landmarks_and_segmentation_inverse_projection.pbtxt");
  DEFINE_SUBGRAPH(PoseLandmarkFiltering, "../../mediapipe/modules/pose_landmark/pose_landmark_filtering.pbtxt");
  DEFINE_SUBGRAPH(PoseLandmarksToRoi, "../../mediapipe/modules/pose_landmark/pose_landmarks_to_roi.pbtxt");
  DEFINE_SUBGRAPH(PoseSegmentationFiltering, "../../mediapipe/modules/pose_landmark/pose_segmentation_filtering.pbtxt");
DEFINE_SUBGRAPH(PoseRendererCpu, "../../mediapipe/graphs/pose_tracking/subgraphs/pose_renderer_cpu.pbtxt");
  DEFINE_SUBGRAPH(PoseLandmarksToRenderData, "../../mediapipe/graphs/pose_tracking/subgraphs/pose_landmarks_to_render_data.pbtxt");

// holistic tracking
DEFINE_SUBGRAPH(HolisticLandmarkCpu, "../../mediapipe/modules/holistic_landmark/holistic_landmark_cpu.pbtxt");
//  DEFINE_SUBGRAPH(PoseLandmarkCpu, "../../mediapipe/modules/pose_landmark/pose_landmark_cpu.pbtxt");
//    DEFINE_SUBGRAPH(PoseDetectionCpu, "../../mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt");
//    DEFINE_SUBGRAPH(PoseDetectionToRoi, "../../mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt");
//    DEFINE_SUBGRAPH(PoseLandmarkByRoiCpu, "../../mediapipe/modules/pose_landmark/pose_landmark_by_roi_cpu.pbtxt");
//      DEFINE_SUBGRAPH(PoseLandmarkModelLoader, "../../mediapipe/modules/pose_landmark/pose_landmark_model_loader.pbtxt");
//      DEFINE_SUBGRAPH(TensorsToPoseLandmarksAndSegmentation, "../../mediapipe/modules/pose_landmark/tensors_to_pose_landmarks_and_segmentation.pbtxt");
//      DEFINE_SUBGRAPH(PoseLandmarksAndSegmentationInverseProjection, "../../mediapipe/modules/pose_landmark/pose_landmarks_and_segmentation_inverse_projection.pbtxt");
//    DEFINE_SUBGRAPH(PoseLandmarkFiltering, "../../mediapipe/modules/pose_landmark/pose_landmark_filtering.pbtxt");
//    DEFINE_SUBGRAPH(PoseLandmarksToRoi, "../../mediapipe/modules/pose_landmark/pose_landmarks_to_roi.pbtxt");
//    DEFINE_SUBGRAPH(PoseSegmentationFiltering, "../../mediapipe/modules/pose_landmark/pose_segmentation_filtering.pbtxt");
  DEFINE_SUBGRAPH(HandLandmarksLeftAndRightCpu, "../../mediapipe/modules/holistic_landmark/hand_landmarks_left_and_right_cpu.pbtxt"); 
   DEFINE_SUBGRAPH(HandLandmarksFromPoseCpu, "../../mediapipe/modules/holistic_landmark/hand_landmarks_from_pose_cpu.pbtxt");
     DEFINE_SUBGRAPH(HandVisibilityFromHandLandmarksFromPose, "../../mediapipe/modules/holistic_landmark/hand_visibility_from_hand_landmarks_from_pose.pbtxt");
     DEFINE_SUBGRAPH(HandLandmarksFromPoseToRecropRoi, "../../mediapipe/modules/holistic_landmark/hand_landmarks_from_pose_to_recrop_roi.pbtxt");
     DEFINE_SUBGRAPH(HandRecropByRoiCpu, "../../mediapipe/modules/holistic_landmark/hand_recrop_by_roi_cpu.pbtxt");
     DEFINE_SUBGRAPH(HandTracking, "../../mediapipe/modules/holistic_landmark/hand_tracking.pbtxt");
       DEFINE_SUBGRAPH(HandLandmarksToRoi, "../../mediapipe/modules/holistic_landmark/hand_landmarks_to_roi.pbtxt");
//     DEFINE_SUBGRAPH(HandLandmarkCpu, "../../mediapipe/modules/hand_landmark/hand_landmark_cpu.pbtxt");
//       DEFINE_SUBGRAPH(HandLandmarkModelLoader, "../../mediapipe/modules/hand_landmark/hand_landmark_model_loader.pbtxt");
  DEFINE_SUBGRAPH(FaceLandmarksFromPoseCpu, "../../mediapipe/modules/holistic_landmark/face_landmarks_from_pose_cpu.pbtxt");
    DEFINE_SUBGRAPH(FaceLandmarksFromPoseToRecropRoi, "../../mediapipe/modules/holistic_landmark/face_landmarks_from_pose_to_recrop_roi.pbtxt");
    DEFINE_SUBGRAPH(FaceDetectionShortRangeByRoiCpu, "../../mediapipe/modules/face_detection/face_detection_short_range_by_roi_cpu.pbtxt");
//      DEFINE_SUBGRAPH(FaceDetectionShortRange, "../../mediapipe/modules/face_detection/face_detection_short_range.pbtxt");
//        DEFINE_SUBGRAPH(FaceDetection, "../../mediapipe/modules/face_detection/face_detection.pbtxt");
    DEFINE_SUBGRAPH(FaceDetectionFrontDetectionsToRoi, "../../mediapipe/modules/holistic_landmark/face_detection_front_detections_to_roi.pbtxt");
    DEFINE_SUBGRAPH(FaceTracking, "../../mediapipe/modules/holistic_landmark/face_tracking.pbtxt");
      DEFINE_SUBGRAPH(FaceLandmarksToRoi, "../../mediapipe/modules/holistic_landmark/face_landmarks_to_roi.pbtxt");
//    DEFINE_SUBGRAPH(FaceLandmarkCpu, "../../mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt");
//      DEFINE_SUBGRAPH(FaceLandmarksModelLoader, "../../mediapipe/modules/face_landmark/face_landmarks_model_loader.pbtxt");
//      DEFINE_SUBGRAPH(TensorsToFaceLandmarks, "../../mediapipe/modules/face_landmark/tensors_to_face_landmarks.pbtxt");
//      DEFINE_SUBGRAPH(TensorsToFaceLandmarksWithAttention, "../../mediapipe/modules/face_landmark/tensors_to_face_landmarks_with_attention.pbtxt");
DEFINE_SUBGRAPH(HolisticTrackingToRenderData, "../../mediapipe/graphs/holistic_tracking/holistic_tracking_to_render_data.pbtxt");
  DEFINE_SUBGRAPH(HandWristForPose, "../../mediapipe/modules/holistic_landmark/hand_wrist_for_pose.pbtxt");

// object tracking
DEFINE_SUBGRAPH(ObjectDetectionSubgraphCpu, "../../mediapipe/graphs/tracking/subgraphs/object_detection_cpu.pbtxt");
DEFINE_SUBGRAPH(ObjectTrackingSubgraphCpu, "../../mediapipe/graphs/tracking/subgraphs/object_tracking_cpu.pbtxt");
  DEFINE_SUBGRAPH(BoxTrackingSubgraphCpu, "../../mediapipe/graphs/tracking/subgraphs/box_tracking_cpu.pbtxt");
DEFINE_SUBGRAPH(RendererSubgraphCpu, "../../mediapipe/graphs/tracking/subgraphs/renderer_cpu.pbtxt");

} // namespace mediapipe

// demo collections
struct {
  char const* demo;
  char const* pbtxt;
} const demo_collections[] = {
  { "face detection", "../../mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt" },
  { "face mesh", "../../mediapipe/graphs/face_mesh/face_mesh_desktop_live.pbtxt" },
  { "hair segmentation", "../../mediapipe/graphs/hair_segmentation/hair_segmentation_desktop_live.pbtxt" },
  { "hand tracking", "../../mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt" },
  { "holistic tracking", "../../mediapipe/graphs/holistic_tracking/holistic_tracking_cpu.pbtxt" },
  { "iris tracking", "../../mediapipe/graphs/iris_tracking/iris_tracking_cpu.pbtxt" },
  { "object tracking", "../../mediapipe/graphs/tracking/object_detection_tracking_desktop_live.pbtxt" },
  { "pose tracking", "../../mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt" },
  { "template matching", "../template_matching/template_matching_desktop_live.pbtxt" },
};
constexpr int total_demos = (int) (sizeof(demo_collections)/sizeof(demo_collections[0]));

constexpr char const* input_stream = "input_video";
constexpr char const* output_stream = "output_video";

int select_graph(mediapipe::CalculatorGraph& graph) {
  printf("\nWhich demo would you like this time? We have...\n");
  for (int i=0; i<total_demos; ++i) {
    printf(" %2d) %s\n", i+1, demo_collections[i].demo);
  }
  printf("\nA: ");
  int select_graph = 0;
  std::cin >> select_graph;

  if (1<=select_graph && select_graph<=total_demos) {
    printf(">> Good choice. moment please...\n\n");
    --select_graph; // 0-base
  } else {
    select_graph = 1; // must < total_demos
    printf(">> I'm sorry? Never mind, I'll give you \'%s\'\n\n", demo_collections[select_graph].demo);
  }

  mediapipe::CalculatorGraphConfig config;
  if (read_config_from_pbtxt(config, demo_collections[select_graph].pbtxt)) {
    download_mediapipe_asset_from_GCS("../mediapipe/modules/face_detection/face_detection_short_range.tflite");
    download_mediapipe_asset_from_GCS("../mediapipe/modules/face_landmark/face_landmark_with_attention.tflite");
    download_mediapipe_asset_from_GCS("../mediapipe/modules/face_landmark/face_landmark.tflite");
    download_mediapipe_asset_from_GCS("../mediapipe/models/hair_segmentation.tflite");
    download_mediapipe_asset_from_GCS("../mediapipe/modules/palm_detection/palm_detection_full.tflite");
    download_mediapipe_asset_from_GCS("../mediapipe/modules/hand_landmark/hand_landmark_full.tflite");
    download_mediapipe_asset_from_GCS("../mediapipe/modules/hand_landmark/handedness.txt");
    download_mediapipe_asset_from_GCS("../mediapipe/modules/pose_landmark/pose_landmark_full.tflite");
    download_mediapipe_asset_from_GCS("../mediapipe/modules/pose_detection/pose_detection.tflite");
    download_mediapipe_asset_from_GCS("../mediapipe/modules/holistic_landmark/hand_recrop.tflite");
    download_mediapipe_asset_from_GCS("../mediapipe/modules/iris_landmark/iris_landmark.tflite");
    download_mediapipe_asset_from_GCS("../mediapipe/models/ssdlite_object_detection.tflite");
    download_mediapipe_asset_from_GCS("../mediapipe/models/ssdlite_object_detection_labelmap.txt");
    download_mediapipe_asset_from_GCS("../mediapipe/modules/pose_landmark/pose_landmark_lite.tflite");
    download_mediapipe_asset_from_GCS("../mediapipe/modules/pose_landmark/pose_landmark_heavy.tflite");
    download_mediapipe_asset_from_GCS("../mediapipe/models/knift_float.tflite");
    download_mediapipe_asset_from_GCS("../mediapipe/models/knift_index.pb");
    download_mediapipe_asset_from_GCS("../mediapipe/models/knift_labelmap.txt");

    absl::Status const status = graph.Initialize(config);
    if (status.ok()) {
      return select_graph;
    } else {
      std::cout << "[Error] " << demo_collections[select_graph].pbtxt << "\n" << status.message() << "\n";
      return -1;
    }
  }

  printf("[Error] %s file not exist or ill-format.\n", demo_collections[select_graph].pbtxt);
  return -2;
}

void mouse_event(int event, int x, int y, int flags, void* user_data) {
  if (user_data) {
    auto& mouse = ((UserInput*) user_data)->mouse;
    mouse.event = event;
    mouse.flags = flags;
    mouse.x = x;
    mouse.y = y;
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  // workaround tflite model path
  {
    char resource_root_dir[128];
    strcpy(resource_root_dir, "--resource_root_dir=../");
    char* argv2[2] = { argv[0], resource_root_dir };
    absl::ParseCommandLine(2, argv2);
  }

  // hello
  printf("Mediapipe LIVE demo start.... [ESC to quit]\n");

  //
  // -input_video=test_video.mp4 -save_video=video_file.mp4
  char const* input_video = nullptr;
  char const* save_video = nullptr;
  for (int i=1; i<argc; ++i) {
    char const* arg = argv[i];
    if (arg) {
      if ('-'==*arg) {
        ++arg;
      }
      if ('-'==*arg) {
        ++arg;
      }
      if (0==memcmp("input_video=", arg, 12)) {
        input_video = arg + 12;
      } else if (0==memcmp("save_video=", arg, 11)) {
        save_video = arg + 11;
      }
    }
  }

  // register option
  register_face_detection_options();

  // construct graph
  mediapipe::CalculatorGraph graph;
  int error = select_graph(graph);
  if (error<0) {
    system("pause");
    return -1;
  }

  char window_title[128];
  sprintf(window_title, "mediepipe - %s", demo_collections[error].demo);

  // need poller get output frame
  auto poller = graph.AddOutputStreamPoller(output_stream);
  if (!poller.ok()) {
    std::cout << "Calculator Graph is loaded, but failed to get output_stream!?\n"
              << poller.status().message() << "\n";
    return -2;
  }

  cv::VideoCapture capture;
  cv::VideoWriter writer;
  if (input_video) {
    printf("Loading video file %s...\n", input_video);
    if (!capture.open(input_video)) {
      printf("Failed to open video file, will try video camera\n");
      input_video = nullptr;
    }
  }

  if (!input_video) {
    printf("Open video camera...\n");
    if (capture.open(0)) {
#if ((CV_MAJOR_VERSION*10+CV_MINOR_VERSION) >= 32)
      capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
      capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
      capture.set(cv::CAP_PROP_FPS, 30);
#endif
    }
  }

  if (!capture.isOpened()) {
    printf("Failed to open video source! App must closed\n");
    return -3;
  }

  printf("\nStart running the calculator graph...\n");
  if (!graph.StartRun({}).ok()) {
    printf("Failed to run Calculator graph!!! Better luck next time.\n");
    return -4;
  }

  // window open up
  cv::namedWindow(window_title, /*flags=WINDOW_AUTOSIZE*/ 1);
  error = 0;

  char text_buf[64];
  auto const time_start = std::chrono::system_clock::now();
  auto fps_update = time_start;
  int frame_count = 0;
  float frame_per_second = 0.0f;

  printf("\nAlright! Start grabbing and processing frames...\n");

  bool const process_key_input = graph.HasInputStream("user_input");
  UserInput user_input;
  if (process_key_input) {
    cv::setMouseCallback(window_title, mouse_event, &user_input);
  }

  for (auto const start_time=std::chrono::system_clock::now();
       user_input.wait_key!=27; user_input.wait_key=cv::waitKeyEx(1)) {
    // close window after wait key
    if (cv::getWindowProperty(window_title, cv::WND_PROP_VISIBLE)<1.0) {
      break;
    }

    // TO-DO: you can actually change demo here...
    if ('1'<=user_input.wait_key && user_input.wait_key<=9) {
    }

    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (!camera_frame_raw.empty()) {
      ++user_input.frame_id;
    } else if (!input_video) {
      printf("Ignore empty frames from camera.\n");
      continue;
    } else {
      printf("Empty frame, end of video reached.\n");
      break;
    }

    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    if (!input_video) {
      cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    }

    // timestamp
    mediapipe::Timestamp const timestamp(get_elapsed_time_microseconds(start_time));

    // process key input
    if (process_key_input) {
      graph.AddPacketToInputStream("user_input", mediapipe::MakePacket<UserInput>(user_input).At(timestamp));

      // reset
      user_input.mouse.event = 0;
      user_input.mouse.flags = 0;
    }

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    if (!graph.AddPacketToInputStream(input_stream, mediapipe::Adopt(input_frame.release()).At(timestamp)).ok()) {
      if (--error<=-5) {
        printf("Failed to send image packet into the graph.\n");
        break;
      } else {
        continue;
      }
    }

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (!poller.value().Next(&packet)) {
      printf("Failed to poll graph result packet.\n");
      error = -6;
      break;
    }

    auto& output_frame = packet.Get<mediapipe::ImageFrame>();

    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

    if (save_video) {
      if (writer.isOpened()) {
        writer.write(output_frame_mat);
      } else {
        writer.open(save_video, cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
                    capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
        if (writer.isOpened()) {
          writer.write(output_frame_mat);
        } else {
          printf("Failed to open save video %s\n", save_video);
          save_video = nullptr;
        }
      }
    }

    // update fps
    ++frame_count;
    auto time_update = get_elapsed_time_microseconds(fps_update);
    if (time_update>1000000) {
      if (frame_count>0) {
        frame_per_second = (float)((1.e+6 * (double)frame_count)/ (double)(time_update+1));
      } else {
        frame_per_second = 0.0f;
      }
      fps_update = std::chrono::system_clock::now();
      frame_count = 0;
    }

    // print message on screen
    sprintf(text_buf, "%dx%d@%.1fHz", output_frame_mat.cols, output_frame_mat.rows, frame_per_second);
    cv::putText(output_frame_mat, text_buf,
                cv::Point(10, 25),     // origin
                cv::FONT_HERSHEY_COMPLEX_SMALL,
                1.25,                  // font scale
                cv::Scalar(255,0,255), // font color, (B,G,R)
                  1,                   // font_thickness
                cv::LINE_AA);          // line type

    // present
    cv::imshow(window_title, output_frame_mat);

    if (0x03==user_input.wait_key) { // capture: Ctrl+C
      cv::imwrite("screen_shot.jpg", output_frame_mat);
    }
  }

  printf("Shutting down...\n");
  cv::destroyAllWindows();
  if (writer.isOpened()) {
    writer.release();
  }
  graph.CloseAllInputStreams();
  graph.WaitUntilDone();

  if (error) {
    system("pause");
  }
  return error;
}