//
// re-write from //mediapipe/mediapipe/examples/desktop/demo_run_graph_main.cc
//
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
#include "mediapipe/modules/face_geometry/protos/face_geometry.pb.h"

constexpr char const* kWindowName = "face landmark with attention";

constexpr char const* kInputStream = "input_video";
constexpr char const* kOutputStream = "output_video";

ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");

absl::Status RunMPPGraph(char const* pbtxt="./face_mesh_desktop_live.pbtxt") {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(pbtxt, &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: " << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;
  bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
  if (load_video) {
    capture.open(absl::GetFlag(FLAGS_input_video_path));
  } else {
    capture.open(0);
  }
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
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_output_video, graph.AddOutputStreamPoller(kOutputStream));

//#define DEBUG_FACE_GEOMETRY_MODULE
#ifdef DEBUG_FACE_GEOMETRY_MODULE
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_multi_face_landmarks, graph.AddOutputStreamPoller("multi_face_landmarks"));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmark_presence, graph.AddOutputStreamPoller("landmark_presence"));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_multi_face_geometry, graph.AddOutputStreamPoller("multi_face_geometry"));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_geometry_presence, graph.AddOutputStreamPoller("geometry_presence"));
  bool debug_no_yet = true;
#endif

  MP_RETURN_IF_ERROR(graph.StartRun({}));

//#define DEBUG_FRAME_ENABLE
#ifdef DEBUG_FRAME_ENABLE
  auto debug_frame = cv::imread("./test_image.bmp");
  load_video = !debug_frame.empty();
#endif

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  while (grab_frames) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
      if (!load_video) {
        LOG(INFO) << "Ignore empty frames from camera.";
        continue;
      }
      LOG(INFO) << "Empty frame, end of video reached.";
      break;
    }

#ifdef DEBUG_FRAME_ENABLE
    if (!debug_frame.empty()) {
      debug_frame.copyTo(camera_frame_raw);
    }
#endif

    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

    if (!load_video) {
      cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    }

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    size_t frame_timestamp_us = (size_t) ((double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6);
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (!poller_output_video.Next(&packet)) break;
    auto& output_frame = packet.Get<mediapipe::ImageFrame>();

    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    if (save_video) {
      if (!writer.isOpened()) {
        LOG(INFO) << "Prepare video writer.";
        writer.open(absl::GetFlag(FLAGS_output_video_path),
                    mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                    capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
        RET_CHECK(writer.isOpened());
      }
      writer.write(output_frame_mat);
    } else {
      cv::imshow(kWindowName, output_frame_mat);
      // Press any key to exit.
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
    }

#ifdef DEBUG_FACE_GEOMETRY_MODULE
    if (debug_no_yet) {
      if (poller_landmark_presence.Next(&packet)) {
        bool const is_present = packet.Get<bool>();
        if (is_present) {
          if (poller_multi_face_landmarks.Next(&packet)) {
#if 0
            auto& output_landmarklist = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
            printf("\n/////\nNormalizedLandmarkList[%d]:\n", (int) output_landmarklist.size());
            for (auto& landmarks: output_landmarklist) {
              int const num_landmarks = landmarks.landmark_size();
              for (int i=0; i<num_landmarks; ++i) {
                auto const& m = landmarks.landmark(i);
                printf(" output: [%d] %f, %f, %f\n", i, m.x(), m.y(), m.z());
              }
            }
            printf("/////\n");
#endif
          }
        }
      }

      if (poller_geometry_presence.Next(&packet)) {
        bool const is_present = packet.Get<bool>();
        if (is_present) {
          if (poller_multi_face_geometry.Next(&packet)) {
            auto const& output_geometry = packet.Get<std::vector<mediapipe::face_geometry::FaceGeometry>>();
            for (auto const& geo : output_geometry) {
              auto const& mesh = geo.mesh();
              int const vertex_buffer_size = mesh.vertex_buffer_size();
              int const index_buffer_size = mesh.index_buffer_size();
              printf("vertex_buffer_size: %d(%d)\n", vertex_buffer_size, vertex_buffer_size/5);
              printf("index_buffer_size: %d(%d)\n", index_buffer_size, index_buffer_size/3);
              auto const& ltm = geo.pose_transform_matrix();
              assert(ltm.packed_data_size()==ltm.rows()*ltm.cols());
              printf("pose_transform_matrix[%d][%d] = {\n", ltm.rows(), ltm.cols());
              auto const* data = ltm.packed_data().data();
              if (mediapipe::MatrixData_Layout_COLUMN_MAJOR==ltm.layout()) { // as default
                printf("  % .6f % .6f % .6f  %.6f\n", data[0], data[4], data[8], data[12]);
                printf("  % .6f % .6f % .6f  %.6f\n", data[1], data[5], data[9], data[13]);
                printf("  % .6f % .6f % .6f  %.6f\n", data[2], data[6], data[10], data[14]);
                printf("  % .6f % .6f % .6f  %.6f\n", data[3], data[7], data[11], data[15]);
              } else {
                printf("  % .6f % .6f % .6f  %.6f\n", data[0], data[1], data[2], data[3]);
                printf("  % .6f % .6f % .6f  %.6f\n", data[4], data[5], data[6], data[7]);
                printf("  % .6f % .6f % .6f  %.6f\n", data[8], data[9], data[10], data[11]);
                printf("  % .6f % .6f % .6f  %.6f\n", data[12], data[13], data[14], data[15]);
              }
              printf("};\n");
            }
          }
        }
      }
      debug_no_yet = false;
    }
#endif
  }

  LOG(INFO) << "Shutting down.";
  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
