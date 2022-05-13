#include "../calculator_graph_util.h"

//
// see discussion https://github.com/google/mediapipe/issues/1162#issuecomment-1123137462
//
// It's NOT pretty, but that's the way mediapipe subgraph system goes...
// to load a calculator graph, you must register all subgraphs, all calculators before hand.
// if you check REGISTER_MEDIAPIPE_GRAPH(), REGISTER_CALCULATOR() and REGISTER_INPUT_STREAM_HANDLER(),
// local static dangling pointers are created to trigger registery work. Linker settings details below...
namespace mediapipe {
// holistic tracking
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
     DEFINE_SUBGRAPH(HandRecropByRoiCpu, "../holistic_tracking/hand_recrop_by_roi_cpu.pbtxt"); // ../../mediapipe/modules/holistic_landmark
     DEFINE_SUBGRAPH(HandTracking, "../../mediapipe/modules/holistic_landmark/hand_tracking.pbtxt");
       DEFINE_SUBGRAPH(HandLandmarksToRoi, "../../mediapipe/modules/holistic_landmark/hand_landmarks_to_roi.pbtxt");
     DEFINE_SUBGRAPH(HandLandmarkCpu, "../hand_tracking/hand_landmark_cpu.pbtxt"); // ../../mediapipe/modules/hand_landmark
       DEFINE_SUBGRAPH(HandLandmarkModelLoader, "../hand_tracking/hand_landmark_model_loader.pbtxt"); // ../../mediapipe/modules/hand_landmark
  DEFINE_SUBGRAPH(FaceLandmarksFromPoseCpu, "../../mediapipe/modules/holistic_landmark/face_landmarks_from_pose_cpu.pbtxt");
    DEFINE_SUBGRAPH(FaceLandmarksFromPoseToRecropRoi, "../../mediapipe/modules/holistic_landmark/face_landmarks_from_pose_to_recrop_roi.pbtxt");
    DEFINE_SUBGRAPH(FaceDetectionShortRangeByRoiCpu, "../holistic_tracking/face_detection_short_range_by_roi_cpu.pbtxt"); // ../../mediapipe/modules/face_detection
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

// face mesh
DEFINE_SUBGRAPH(FaceLandmarkFrontCpu, "../../mediapipe/modules/face_landmark/face_landmark_front_cpu.pbtxt");
  DEFINE_SUBGRAPH(FaceDetectionShortRangeCpu, "../face_detection/face_detection_short_range_cpu.pbtxt"); // ../../mediapipe/modules/face_detection
  //DEFINE_SUBGRAPH(FaceDetectionShortRangeCommon, "../../mediapipe/modules/face_detection/face_detection_short_range_common.pbtxt");
  DEFINE_SUBGRAPH(FaceDetectionFrontDetectionToRoi, "../../mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt");
//DEFINE_SUBGRAPH(FaceLandmarkCpu, "../../mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt");
  //DEFINE_SUBGRAPH(FaceLandmarksModelLoader, "../face_mesh/face_landmarks_model_loader.pbtxt"); // ../../mediapipe/modules/face_landmark
  //DEFINE_SUBGRAPH(TensorsToFaceLandmarks, "../../mediapipe/modules/face_landmark/tensors_to_face_landmarks.pbtxt");
  //DEFINE_SUBGRAPH(TensorsToFaceLandmarksWithAttention, "../../mediapipe/modules/face_landmark/tensors_to_face_landmarks_with_attention.pbtxt");
  DEFINE_SUBGRAPH(FaceLandmarkLandmarksToRoi, "../../mediapipe/modules/face_landmark/face_landmark_landmarks_to_roi.pbtxt");
DEFINE_SUBGRAPH(FaceRendererCpu, "../../mediapipe/graphs/face_mesh/subgraphs/face_renderer_cpu.pbtxt");

// face detection
//DEFINE_SUBGRAPH(FaceDetectionShortRangeCpu, "../face_detection/face_detection_short_range_cpu.pbtxt");  // ../../mediapipe/modules/face_detection
  //DEFINE_SUBGRAPH(FaceDetectionShortRangeCommon, "../../mediapipe/modules/face_detection/face_detection_short_range_common.pbtxt");

// hand tracking
DEFINE_SUBGRAPH(HandLandmarkTrackingCpu, "../../mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.pbtxt");
  DEFINE_SUBGRAPH(PalmDetectionCpu, "../../mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt");
    DEFINE_SUBGRAPH(PalmDetectionModelLoader, "../hand_tracking/palm_detection_model_loader.pbtxt"); //../../mediapipe/modules/palm_detection 
  DEFINE_SUBGRAPH(PalmDetectionDetectionToRoi, "../../mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt");
//DEFINE_SUBGRAPH(HandLandmarkCpu, "../hand_tracking/hand_landmark_cpu.pbtxt"); // ../../mediapipe/modules/hand_landmark
  //DEFINE_SUBGRAPH(HandLandmarkModelLoader, "../hand_tracking/hand_landmark_model_loader.pbtxt"); // ../../mediapipe/modules/hand_landmark
  DEFINE_SUBGRAPH(HandLandmarkLandmarksToRoi, "../../mediapipe/modules/hand_landmark/hand_landmark_landmarks_to_roi.pbtxt");
DEFINE_SUBGRAPH(HandRendererSubgraph, "../../mediapipe/graphs/hand_tracking/subgraphs/hand_renderer_cpu.pbtxt");

// iris tracking
//DEFINE_SUBGRAPH(FaceLandmarkFrontCpu, "../../mediapipe/modules/face_landmark/face_landmark_front_cpu.pbtxt");
//  DEFINE_SUBGRAPH(FaceDetectionShortRangeCpu, "../face_detection/face_detection_short_range_cpu.pbtxt");
//    DEFINE_SUBGRAPH(FaceDetectionShortRangeCommon, "../../mediapipe/modules/face_detection/face_detection_short_range_common.pbtxt");
//  DEFINE_SUBGRAPH(FaceDetectionFrontDetectionToRoi, "../../mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt");
//  DEFINE_SUBGRAPH(FaceLandmarkCpu, "../../mediapipe/modules/face_landmark/face_landmark_cpu.pbtxt");
//    DEFINE_SUBGRAPH(FaceLandmarksModelLoader, "../face_mesh/face_landmarks_model_loader.pbtxt");
//    DEFINE_SUBGRAPH(TensorsToFaceLandmarks, "../../mediapipe/modules/face_landmark/tensors_to_face_landmarks.pbtxt");
//    DEFINE_SUBGRAPH(TensorsToFaceLandmarksWithAttention, "../../mediapipe/modules/face_landmark/tensors_to_face_landmarks_with_attention.pbtxt");
//  DEFINE_SUBGRAPH(FaceLandmarkLandmarksToRoi, "../../mediapipe/modules/face_landmark/face_landmark_landmarks_to_roi.pbtxt");
DEFINE_SUBGRAPH(IrisLandmarkLeftAndRightCpu, "../../mediapipe/modules/iris_landmark/iris_landmark_left_and_right_cpu.pbtxt");
  DEFINE_SUBGRAPH(IrisLandmarkLandmarksToRoi, "../../mediapipe/modules/iris_landmark/iris_landmark_landmarks_to_roi.pbtxt");
  DEFINE_SUBGRAPH(IrisLandmarkCpu, "../iris_tracking/iris_landmark_cpu.pbtxt");  // ../../mediapipe/modules/iris_landmark
DEFINE_SUBGRAPH(IrisRendererCpu, "../../mediapipe/graphs/iris_tracking/subgraphs/iris_renderer_cpu.pbtxt");

// object tracking
DEFINE_SUBGRAPH(ObjectDetectionSubgraphCpu, "../object_tracking/object_detection_cpu.pbtxt");
DEFINE_SUBGRAPH(ObjectTrackingSubgraphCpu, "../../mediapipe/graphs/tracking/subgraphs/object_tracking_cpu.pbtxt");
  DEFINE_SUBGRAPH(BoxTrackingSubgraphCpu, "../../mediapipe/graphs/tracking/subgraphs/box_tracking_cpu.pbtxt");
DEFINE_SUBGRAPH(RendererSubgraphCpu, "../../mediapipe/graphs/tracking/subgraphs/renderer_cpu.pbtxt");

// pose tracking
//DEFINE_SUBGRAPH(PoseLandmarkCpu, "../../mediapipe/modules/pose_landmark/pose_landmark_cpu.pbtxt");
//  DEFINE_SUBGRAPH(PoseDetectionCpu, "../pose_tracking/pose_detection_cpu.pbtxt"); // ../../mediapipe/modules/pose_detection
//  DEFINE_SUBGRAPH(PoseDetectionToRoi, "../../mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt");
//  DEFINE_SUBGRAPH(PoseLandmarkByRoiCpu, "../../mediapipe/modules/pose_landmark/pose_landmark_by_roi_cpu.pbtxt");
//    DEFINE_SUBGRAPH(PoseLandmarkModelLoader, "../pose_tracking/pose_landmark_model_loader.pbtxt"); // ../../mediapipe/modules/pose_landmark
//    DEFINE_SUBGRAPH(TensorsToPoseLandmarksAndSegmentation, "../../mediapipe/modules/pose_landmark/tensors_to_pose_landmarks_and_segmentation.pbtxt");
//    DEFINE_SUBGRAPH(PoseLandmarksAndSegmentationInverseProjection, "../../mediapipe/modules/pose_landmark/pose_landmarks_and_segmentation_inverse_projection.pbtxt");
//  DEFINE_SUBGRAPH(PoseLandmarkFiltering, "../../mediapipe/modules/pose_landmark/pose_landmark_filtering.pbtxt");
//  DEFINE_SUBGRAPH(PoseLandmarksToRoi, "../../mediapipe/modules/pose_landmark/pose_landmarks_to_roi.pbtxt");
//  DEFINE_SUBGRAPH(PoseSegmentationFiltering, "../../mediapipe/modules/pose_landmark/pose_segmentation_filtering.pbtxt");
DEFINE_SUBGRAPH(PoseRendererCpu, "../../mediapipe/graphs/pose_tracking/subgraphs/pose_renderer_cpu.pbtxt");

} // namespace mediapipe

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
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

constexpr char const* input_stream = "input_video";
constexpr char const* output_stream = "output_video";

// demo collections
struct {
  char const* demo;
  char const* pbtxt;
} const demo_collections[] = {
  { "face detection", "../../mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt" },
  { "face mesh", "../../mediapipe/graphs/face_mesh/face_mesh_desktop_live.pbtxt" },
  { "hair segmentation", "../hair_segmentation/hair_segmentation_desktop_live.pbtxt" },
  { "hand tracking", "../../mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt" },
  { "holistic tracking", "../../mediapipe/graphs/holistic_tracking/holistic_tracking_cpu.pbtxt" },
  { "iris tracking", "../../mediapipe/graphs/iris_tracking/iris_tracking_cpu.pbtxt" },
  { "object tracking", "../../mediapipe/graphs/tracking/object_detection_tracking_desktop_live.pbtxt" },
  { "pose tracking", "../../mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt" },
  { "template matching (i don't really know what is this!?)", "../template_matching/template_matching_desktop_live.pbtxt" },
};
constexpr int total_demos = (int) (sizeof(demo_collections)/sizeof(demo_collections[0]));

// select graph
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
    if (graph.Initialize(config).ok()) {
      return select_graph;
    } else {
      printf("[Error] %s \n", demo_collections[select_graph].pbtxt);
      return -1;
    }
  }

  printf("[Error] %s \n", demo_collections[select_graph].pbtxt);
  return -2;
}

inline int64_t get_elapsed_time_microseconds(std::chrono::time_point<std::chrono::system_clock> const& start_time) {
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now()-start_time).count();
}

int main(int argc, char** argv) {
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
    printf("Calculator Graph is loaded, but failed to get output_stream!?\n");
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
  for (int key(-1),frame_id(-1); key!=27; key=cv::waitKeyEx(1)) {

    // TO-DO: you can actually change demo here...
    if ('1'<=key && key<=9) {
    }

    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (!camera_frame_raw.empty()) {
      ++frame_id;
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

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    if (!graph.AddPacketToInputStream(input_stream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(get_elapsed_time_microseconds(time_start)))).ok()) {
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
        writer.open(save_video,
                    cv::VideoWriter::fourcc('a', 'v', 'c', '1'),  // .mp4
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
  }

  printf("Shutting down...\n");
  cv::destroyAllWindows();
  if (writer.isOpened()) {
    writer.release();
  }
  graph.CloseAllInputStreams();
  graph.WaitUntilDone();

  return error;
}