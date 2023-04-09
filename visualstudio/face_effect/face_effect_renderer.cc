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
//
// node {
//   calculator: "FaceEffectRenderer"
//   input_side_packet: "FACE_GEOMETRY:face_geometry_path"
//   input_side_packet: "MODEL_PATH:model_path"
//   input_stream: "USER_INPUT:user_input"
//   input_stream: "NORM_LANDMARKS:multi_face_landmarks"
//   input_stream: "IMAGE:throttled_input_video"
//   output_stream: "IMAGE:output_video"
// }
//
#include "../calculator_graph_util.h"
#include "procrustes_solver.h"
#include "opengl_renderer.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/formats/landmark.pb.h"

// compare with face_geometry module
#include "mediapipe/modules/face_geometry/protos/face_geometry.pb.h"
#include "mediapipe/modules/face_geometry/geometry_pipeline_calculator.pb.h"

namespace {

constexpr char const* kImageTag = "IMAGE";
constexpr char const* kNormLandmarksTag = "NORM_LANDMARKS";
constexpr char const* kUserInputTag = "USER_INPUT";
constexpr char const* kMultiFaceGeometryTag = "MULTI_FACE_GEOMETRY"; // optional, check face_effect.cc

constexpr int num_effect_meshes = 3;

void print_help() {
  printf("\n[INFO]\nface_effect key controls:\n");
  printf("    Mouse L+Move: 3D view angle\n");
  printf("          Effect: '1', '2', '3' or SPACEBAR\n");
  printf("       Primitive: 'm' for triangles, lines or points\n");
  printf(" Canonical Model: 'k'\n");
  printf(" Export OBJ file: 'o'\n");
  printf("      Camera FOV: 'a' or 's'\n");
  printf("     Model Scale: 'z' or 'x'\n");
  printf("            HELP: 'h'\n\n");
}

} // namespace

namespace mediapipe {

class FaceEffectRenderer : public CalculatorBase, private OpenGLRenderer {
  // canonical face model vertices
  std::vector<Vector3> canonical_face_model_;

  // to solve landmarks from image space to metric space
  WeightedOrthogonalProblemSolver wop_solver_;
  std::vector<int> procrustes_landmark_indices_;

  std::vector<int> triangle_list_;

  // effect to show
  DrawMesh effect_meshes_[num_effect_meshes];
  DrawMesh face_mesh_;

  // bring calculator contex to render thread
  CalculatorContext* calc_ctx_{nullptr};

  // a render task - [to be removed]
  struct {
    std::vector<NormalizedLandmarkList> const* landmark_set{nullptr};
    uint8 const* background{nullptr}; // camera image as background, RGB8 format
    uint8 const* result{nullptr}; // GL render result, RGB8 format
    int width{0}, height{0};

    operator bool() const {
      return background && width>0 && height>0;
    }

    void Reset() {
      landmark_set = nullptr;
      result = background = nullptr;
      width = height = 0;
    }
  } render_task_;

  
  float print_pose_transform_error_threshold_{0.01f};

  // tweakable...
  float camera_vertical_fov_{43.0f};
  int mouse_x_{0}, mouse_y_{0};
  uint8_t canonical_face_model_scale_pct_{100};
  uint8_t current_effect_{0};
  uint8_t draw_mode_:2;
  uint8_t view_angle_3d_:1;
  uint8_t show_canonical_landmarks_:2;
  uint8_t trigger_save_OBJ_file_:1,:2;

  //
  // solve face landmark from weak perspective projection model to 3D metric space
  bool SolveFaceLandmark3D_(Matrix3& transform, Vector3 landmarks[478],
                            NormalizedLandmarkList const& face_landmarks) {
    //
    // From mediapipe...
    // "The Face Landmark Model performs a single-camera face landmark detection
    //  in the screen coordinate space: the X- and Y- coordinates are normalized
    //  screen coordinates, while the Z coordinate is relative and is scaled as
    //  the X coodinate under the weak perspective projection camera model."
    //  https://developers.googleblog.com/2020/09/mediapipe-3d-face-transform.html
    //
    // To convert landmarks to metric 3D space from screen coordinate weak perspective
    // projection space....
    //
    // 1) convert the X- and Y- coordinates from pixel coordinate to projection plane.
    // This scaling factor s must apply on z-coordinate as 'Z coordinate is relative and
    // is scaled as the X coordinate'.
    //
    // 2) now the Z coordinate is with the same scale as X- and Y- coordinate (But
    // may need to offset). Suppose (X, Y, Z) is the actual point in 3D metric space
    // (before weak perspective projection applied) and Zave is the average constant depth.
    // (see https://en.wikipedia.org/wiki/3D_projection#Weak_perspective_projection)
    // Giving Px and Py from the first step, the weak perspective projection formula,
    //    Px = X / Zave
    //    Py = Y / Zave
    // gives us the idea that after the first step, Zave is should be 1.0.
    // So having z-coordinate the right scale, we shall offset Z to make Z average 1.0.
    //
    // 3) for now, the remain question is find the z_scale to unproject all landmarks.
    // (with byproduct of canonincal mesh transformation)
    //

    // num_landmarks should be 468 or 478 with_attention
    int const num_landmarks = face_landmarks.landmark_size();
    assert(468==num_landmarks || 478==num_landmarks);

    // 1) scale normalized landmarks to make X- Y- coordinate on the projection plane (Z=1.0).
    float const y_scale = 2.0f*tan(0.5f*camera_vertical_fov_*0.017453293f);
    float const x_scale = y_scale*(float)render_task_.width/(float)render_task_.height;
    float z_scale = -x_scale; // z has the same scale with x, negate to convert depth value to OpenGL Z coordinate
    float z_avg = 0.0f;
    for (int i=0; i<num_landmarks; ++i) {
      NormalizedLandmark const& m = face_landmarks.landmark(i);
      auto& p = landmarks[i];
      p.x = x_scale*(m.x() - 0.5f); // OpenGL X = Right
      p.y = y_scale*(0.5f - m.y()); // OpenGL Y = Up
      z_avg += p.z = z_scale*m.z(); // OpenGL Z = Toward Camera
    }
    z_avg /= (float) num_landmarks;

    // 2) with all Zs in correct scale, shift Z so that the average of Z on projection plane
    //    in OpenGL coordinate system, projection plane at Z=-1
    float const z_offset = -1.0f - z_avg;
    int const procrustes_landmark_basis_size = (int) procrustes_landmark_indices_.size();
    std::vector<Vector3> targets(procrustes_landmark_basis_size);
    for (int i=0; i<procrustes_landmark_basis_size; ++i) {
      (targets[i] = landmarks[procrustes_landmark_indices_[i]]).z += z_offset;
    }

    // 3) 1st pass calculate the rough scale factor
    float canonical_to_targets_scale = wop_solver_.Solve(nullptr, targets.data());

    //
    // use this rough scale factor to stretch targets model (like Procrustes).
    // The aim is to make target model size close to canonical face model while keeping
    // landmarks' pixel coordinates not change.
    z_scale = 1.0f/canonical_to_targets_scale; // targets_to_canonical_scale
    for (int i=0; i<procrustes_landmark_basis_size; ++i) {
      auto& t = targets[i];
      t.z *= z_scale;
      t.x *= -t.z; // unproject, t.z<0.0f
      t.y *= -t.z;
    }

    // now, landmarks are in 3D camera space (NOT weak perspective projection space)

    // the 2nd pass, again find scale difference. aggregate scale factor
    canonical_to_targets_scale *= z_scale = wop_solver_.Solve(nullptr, targets.data());

    // with all landmarks in camera space, simply do further scaling.
    z_scale = 1.0f/z_scale;
    for (int i=0; i<procrustes_landmark_basis_size; ++i) {
      auto& t = targets[i];
      t.z *= z_scale; // scale
      t.x *= z_scale;
      t.y *= z_scale;
    }

    // final pass gives us the final scale factor to solve 3D metric space landmarks
    // (in pin-hole camera projection model space) with the best-fit canonical face model transformation
    canonical_to_targets_scale *= wop_solver_.Solve(&transform, targets.data());

    z_scale = 1.0f/canonical_to_targets_scale;
    for (int i=0; i<num_landmarks; ++i) {
      auto& v = landmarks[i];
      v.z = (v.z + z_offset)*z_scale;
      v.x *= -v.z; // unproj, t.z<0.0f
      v.y *= -v.z;
    }

    //
    // the precision of landmarks[] is dependent on 2 factors
    // 1) camera projection model such as:
    //     a) estimated vertical field of view (use key 'a'/'s' to try out)
    //     b) distortion correction (not implemnted)
    //
    // 2) the size different between your face and canonical face model. (use key 'z'/'x' to adjust)
    //    the distance between 2 eyes (landmarks #133 to #362) of the canonical model = 3.7cm.
    //    refer:
    //     a) canonical face mesh: (text files)
    //        mediapipe/modules/face_geometry/data/geometry_pipeline_metadata_landmarks.pbtxt
    //        mediapipe/modules/face_geometry/data/canonical_face_model.obj
    //
    //     b) landmark indices:
    //        mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    //
    // anyway, stay in the image center to get best effect!
    //

    return true;
  }

 public:
  FaceEffectRenderer() = default;
  ~FaceEffectRenderer() override = default;

  static abslx::Status GetContract(CalculatorContract* cc) {
    int input_checks = 0;

    // input side packets
    auto& input_side_packets = cc->InputSidePackets();
    for (CollectionItemId id=input_side_packets.BeginId(); id<input_side_packets.EndId(); ++id) {
      auto tag_and_index = input_side_packets.TagAndIndexFromId(id);
      auto& packet = input_side_packets.Get(id);
      if ("CAMERA_FOV"==tag_and_index.first) {
        packet.Set<float>();
      } else if ("FACE_GEOMETRY"==tag_and_index.first) {
        packet.Set<std::string>();
        input_checks |= 1;
      } else if ("MODEL_PATH"==tag_and_index.first) {
        packet.Set<std::string>();
        input_checks |= 2;
      }
    }

    // Data streams to render.
    auto& inputs = cc->Inputs();
    auto& outputs = cc->Outputs();
    for (CollectionItemId id=inputs.BeginId(); id<inputs.EndId(); ++id) {
      auto tag_and_index = inputs.TagAndIndexFromId(id);
      auto& type = inputs.Get(id);
      if (kUserInputTag==tag_and_index.first) {
        type.Set<UserInput>();
        input_checks |= 4;
      } else if (kNormLandmarksTag==tag_and_index.first) {
        type.Set<std::vector<NormalizedLandmarkList>>();
        input_checks |= 8;
      } else if (kImageTag==tag_and_index.first) {
        type.Set<ImageFrame>();
        if (outputs.HasTag(kImageTag)) {
          outputs.Tag(kImageTag).Set<ImageFrame>();
          input_checks |= 16;
        }
      } else if (kMultiFaceGeometryTag==tag_and_index.first) {
        type.Set<std::vector<face_geometry::FaceGeometry>>();
      }
    }

    if (31==input_checks) {
      return abslx::OkStatus();
    }

    return abslx::InternalError("FaceEffectRenderer::GetContract() - illegal parameters");
  }

  abslx::Status Open(CalculatorContext* cc) override {
    if (!cc->Inputs().HasTag(kImageTag)) {
      return abslx::InternalError("FaceEffectRenderer::Open() - no image frame input not available");
    }
    calc_ctx_ = cc; // OnInitedGL_()
    OpenGLRenderer::Run_();

    cc->SetOffset(TimestampDiff(0));

    draw_mode_ = 0;
    view_angle_3d_ = false;
    show_canonical_landmarks_ = 0;
    trigger_save_OBJ_file_ = false;
    print_pose_transform_error_threshold_ = 0.01f;

    print_help();

    return abslx::OkStatus();
  }

  abslx::Status Process(CalculatorContext* cc) override {
    auto const& inputs = cc->Inputs();

    // user input
    if (inputs.HasTag(kUserInputTag) && !inputs.Tag(kUserInputTag).IsEmpty()) {
      auto const& input = inputs.Tag(kUserInputTag).Get<UserInput>();

      // key
      if (' '==input.wait_key) {
        current_effect_ = (current_effect_+1)%num_effect_meshes;
        printf("[INFO] Effect= %d\n", (int) current_effect_);
      } else if ('1'<=input.wait_key && input.wait_key<='1'+num_effect_meshes) {
        current_effect_ = input.wait_key - '1';
        printf("[INFO] Effect= %d\n", (int) current_effect_);
      } else if ('m'==input.wait_key) {
        //draw_mode_ = (2==draw_mode_) ? 0:2; // 0:triangles, 2:lines
        draw_mode_ = (draw_mode_+1)%3;
        char const* types[] = { "triangle", "point", "line" };
        printf("[INFO] Primitive type= %s\n", types[draw_mode_]);
      } else if ('z'==input.wait_key) {
        if (canonical_face_model_scale_pct_>50) {
          --canonical_face_model_scale_pct_;
          printf("[INFO] Model Scale= %d%%\n", (int) canonical_face_model_scale_pct_);
        }
      } else if ('x'==input.wait_key) {
        if (canonical_face_model_scale_pct_<200) {
          ++canonical_face_model_scale_pct_;
          printf("[INFO] Model Scale= %d%%\n", (int) canonical_face_model_scale_pct_);
        }
      } else if ('a'==input.wait_key) {
        if (camera_vertical_fov_>30.0f) {
          camera_vertical_fov_ -= 1.0f;
          printf("[INFO] Camera FOV= %.1f\n", camera_vertical_fov_);
          OpenGLRenderer::SetCameraPerspective(camera_vertical_fov_);
        }
      } else if ('s'==input.wait_key) {
        if (camera_vertical_fov_<120) {
          camera_vertical_fov_ += 1.0f;
          printf("[INFO] Camera FOV= %.1f\n", camera_vertical_fov_);
          OpenGLRenderer::SetCameraPerspective(camera_vertical_fov_);
        }
      } else if ('o'==input.wait_key) {
        trigger_save_OBJ_file_ = true;
      } else if ('k'==input.wait_key) {
        show_canonical_landmarks_ = (show_canonical_landmarks_+1)%3;
      } else if ('e'==input.wait_key) {
        print_pose_transform_error_threshold_ = 0.01f; // reset threshold
      } else if ('h'==input.wait_key) {
        print_help();
      } else if (input.wait_key>0) {
        printf("[INFO] key %08X not handled\n", input.wait_key);
      }

      // mouse
      mouse_x_ = input.mouse.x;
      mouse_y_ = input.mouse.y;
      if (cv::EVENT_LBUTTONDOWN==input.mouse.event) {
        //printf("[INFO] MOUSE LBUTTONDOWN @ %d, %d\n", input.mouse.x, input.mouse.y);
        view_angle_3d_ = true;
      } else if (cv::EVENT_LBUTTONUP==input.mouse.event) {
        //printf("[INFO] MOUSE LBUTTONUP @ %d, %d\n", input.mouse.x, input.mouse.y);
        view_angle_3d_ = false;
      } else {
        if (cv::EVENT_FLAG_LBUTTON&input.mouse.flags) {
          view_angle_3d_ = true;
          //assert(1==mouse_LButton_down_);
          //printf("[INFO] MOUSE L+MOVE @ %d, %d\n", input.mouse.x, input.mouse.y);
        } else {
          //view_angle_3d_ = false;
          //assert(0==mouse_LButton_down_);
          //printf("[INFO] MOUSE event:%d flags:0x%X @ %d, %d\n", input.mouse.event, input.mouse.flags, input.mouse.x, input.mouse.y);
        }
      }
    }

    auto& outputs = cc->Outputs();
    if (outputs.HasTag(kImageTag) && inputs.HasTag(kImageTag) && !inputs.Tag(kImageTag).IsEmpty()) {
      auto const& input_frame = inputs.Tag(kImageTag).Get<ImageFrame>();
      ImageFormat::Format const target_format = input_frame.Format();
      assert(ImageFormat::SRGB==target_format);

      // init render task...
      calc_ctx_ = cc;
      render_task_.background = const_cast<uint8*>(input_frame.PixelData());
      render_task_.result = nullptr;
      render_task_.width = input_frame.Width();
      render_task_.height = input_frame.Height();
      if (inputs.HasTag(kNormLandmarksTag) && !inputs.Tag(kNormLandmarksTag).IsEmpty()) {
        render_task_.landmark_set = &(inputs.Tag(kNormLandmarksTag).Get<std::vector<NormalizedLandmarkList>>());
      } else {
        render_task_.landmark_set = nullptr;
      }

      // kick off rendering
      OpenGLRenderer::FrameMove_();

      if (render_task_.result) {
        outputs.Tag(kImageTag).Add(new ImageFrame(target_format,
                                   render_task_.width, render_task_.height, render_task_.width*3,
                                   (uint8_t*) render_task_.result,
                                   ImageFrame::PixelDataDeleter::kNone),
                                   cc->InputTimestamp());
      } else {
        // copy background image
        cv::Mat render_frame(render_task_.height, render_task_.width, CV_8UC3);
        formats::MatView(&input_frame).copyTo(render_frame);

        // draw landmarks
        if (inputs.HasTag(kNormLandmarksTag) && !inputs.Tag(kNormLandmarksTag).IsEmpty()) {
          cv::Point p;
          auto const& landmarks_set = inputs.Tag(kNormLandmarksTag).Get<std::vector<NormalizedLandmarkList>>();
          for (auto const& face_landmarks: landmarks_set) {
            int const num_landmarks = face_landmarks.landmark_size();
            for (int i=0; i<num_landmarks; ++i) {
              NormalizedLandmark const& m = face_landmarks.landmark(i);
              p.x = (int) (m.x()*(float)render_task_.width + 0.5f);
              p.y = (int) (m.y()*(float)render_task_.height + 0.5f);
              cv::circle(render_frame, p, 3, cv::Scalar(0,255,0), -1);
            }
          }
        }

        // copy result to output
        auto output_frame = abslx::make_unique<ImageFrame>(target_format, render_task_.width, render_task_.height);
        output_frame->CopyPixelData(target_format, render_task_.width, render_task_.height, render_frame.data,
                                    ImageFrame::kDefaultAlignmentBoundary);
        outputs.Tag(kImageTag).Add(output_frame.release(), cc->InputTimestamp());
      }
    }

    return abslx::OkStatus();
  }

  abslx::Status Close(CalculatorContext* cc) override {
    render_task_.Reset();
    OpenGLRenderer::Stop_();
    return abslx::OkStatus();
  }

  // OpenGL Renderer
  bool OnInitedGL_() override {
    if (nullptr==calc_ctx_) {
      return false;
    }

    Mesh mesh;
    char const* model_path = nullptr;
    auto& input_side_packets = calc_ctx_->InputSidePackets();
    for (CollectionItemId id=input_side_packets.BeginId(); id<input_side_packets.EndId(); ++id) {
      auto tag_and_index = input_side_packets.TagAndIndexFromId(id);
      auto& packet = input_side_packets.Get(id);
      if ("CAMERA_FOV"==tag_and_index.first) {
        camera_vertical_fov_ = packet.Get<float>();
      } else if ("FACE_GEOMETRY"==tag_and_index.first) {
        if (!wop_solver_.ReadFromFile(mesh, procrustes_landmark_indices_, packet.Get<std::string>().c_str())) {
          break;
        }
      } else if ("MODEL_PATH"==tag_and_index.first) {
        model_path = packet.Get<std::string>().c_str();
      }
    }

    if (model_path && !procrustes_landmark_indices_.empty()) {
      current_effect_ = 0;
      if (camera_vertical_fov_<0.0f) {
        camera_vertical_fov_ = 43.0f;
      }
      OpenGLRenderer::SetCameraPerspective(camera_vertical_fov_, 1.0f, 100.0f);

      // save indices (for saving obj files)
      triangle_list_ = mesh.indices;

      //
      // [0] : glasses.pbtxt + glasses.pngblob
      // [1] : axis.pbtxt + axis.pngblob
      // [2] : canonical face mesh + facepaint.pngblob
      char filename[256];
      sprintf(filename, "%s/facepaint.pngblob", model_path);
      if (effect_meshes_[2].Create(mesh, filename)) {
        canonical_face_model_.clear();
        canonical_face_model_.reserve(mesh.vertices.size());
        for (auto const& v:mesh.vertices) {
          canonical_face_model_.push_back({v.x, v.y, v.z});
        }
        face_mesh_.Create(mesh, nullptr);

        sprintf(filename, "%s/glasses.pbtxt", model_path);
        if (LoadMeshFrom_pbtxt(mesh, filename)) {
          sprintf(filename, "%s/glasses.pngblob", model_path);
          if (effect_meshes_[0].Create(mesh, filename)) {
            sprintf(filename, "%s/axis.pbtxt", model_path);
            if (LoadMeshFrom_pbtxt(mesh, filename)) {
              sprintf(filename, "%s/axis.pngblob", model_path);
              if (effect_meshes_[1].Create(mesh, filename)) {
                return true;
              }
            }
          }
        }
      }
    }
    return false;
  }
  bool DrawFrame_() override {
    if (render_task_ &&
        OpenGLRenderer::BeginScene(view_angle_3d_ ? nullptr:render_task_.background, render_task_.width, render_task_.height)) {
      // draw each landmarks
      if (render_task_.landmark_set) {
        Vector3 landmarks[478]; // landmarks in metric 3D space
        Vector2 texcoords[478]; // texture coordinates (normalized_landmarks)
        Matrix3 xform; // transform canonical face model to landmarks in metric 3D space
        Matrix3 view;  // view transform as identity

        if (view_angle_3d_) {
          float yaw = 0.5f - (float)mouse_x_/(float)render_task_.width;
          float pitch = (float)mouse_y_/(float)render_task_.height - 0.5f;
          float ca = cos(pitch);
          float sa = sin(pitch);

          float cb = cos(yaw);
          float sb = sin(yaw);

          view.m11 = cb;   view.m12 = sb*sa; view.m13 = sb*ca;
          view.m21 = 0.0f; view.m22 = ca;    view.m23 = -sa;
          view.m31 = -sb;  view.m32 = cb*sa; view.m33 = cb*ca;

          constexpr float dist = 35.0f;
          view.m14 = view.m13*dist;
          view.m24 = view.m23*dist;
          view.m34 = view.m33*dist - dist;
        }

        face_geometry::FaceGeometry const* geometry_list = nullptr;
        if (calc_ctx_->Inputs().HasTag(kMultiFaceGeometryTag)) {
          auto const& packet = calc_ctx_->Inputs().Tag(kMultiFaceGeometryTag);
          if (!packet.IsEmpty()) {
            geometry_list = (packet.Get<std::vector<face_geometry::FaceGeometry>>()).data();
          }
        }

        auto& face_paint = effect_meshes_[2];
        auto& current_fx = effect_meshes_[current_effect_%num_effect_meshes];
        for (auto const& normalized_landmarks: *render_task_.landmark_set) {
          int const num_landmarks = normalized_landmarks.landmark_size();
          if (SolveFaceLandmark3D_(xform, landmarks, normalized_landmarks)) {
            //
            // export Wavefront .obj file
            if (trigger_save_OBJ_file_) {
              time_t rawtime;
              time(&rawtime);
              struct tm* time = localtime(&rawtime);
              char filename[64];
              if (time) {
                sprintf(filename, "%4d%02d%02dT%02d%02d%02d",
                        time->tm_year + 1900, time->tm_mon + 1,  time->tm_mday,
                        time->tm_hour, time->tm_min, time->tm_sec);
              } else {
                static int s_obj_file_id = 0; // no worry
                sprintf(filename, "%06d", ++s_obj_file_id);
              }

              char fullpath[64];
              sprintf(fullpath, "./obj_out/%s.obj", filename);
              printf("exporting obj file => '%s'\n", fullpath);
              FILE* file = fopen(fullpath, "wb");
              if (file) {
                int const total_triangles = (int) triangle_list_.size()/3;

                fprintf(file, "# face mesh OBJ exporter by andre.hl.chen@gmail.com 2022\n");
                if (time) {
                  fprintf(file, "# File created %d/%02d/%02d %02d:%02d:%02d\n",
                          time->tm_year + 1900, time->tm_mon + 1,  time->tm_mday,
                          time->tm_hour, time->tm_min, time->tm_sec);
                }
                fprintf(file, "# reference: modules/face_geometry/data/canonical_face_model.obj\n\n");

                fprintf(file, "mtllib %s.mtl\n\n", filename);
                fprintf(file, "#\n# 468 vertices, %d triangles\n#\n\n", total_triangles);
                fprintf(file, "usemtl face_from_camera\n");

                auto const& v0 = landmarks[4];
                for (int i=0; i<468; ++i) {
                  auto const& v = landmarks[i];
                  fprintf(file, "v %.4f %.4f %.4f\n", v.x-v0.x, v.y-v0.y, v.z-v0.z);
                }
                for (int i=0; i<468; ++i) {
                  auto const& vt = normalized_landmarks.landmark(i);
                  fprintf(file, "vt %.4f %.4f\n", vt.x(), 1.0f-vt.y());
                }
                //
                // TO-DO : export normal vectors, vn
                //

                int const* indices = triangle_list_.data();
                int a, b, c; // obj file uses 1-index
                for (int i=0; i<total_triangles; ++i) {
                  a = (*indices++) + 1;
                  b = (*indices++) + 1;
                  c = (*indices++) + 1;
                  fprintf(file, "f %d/%d %d/%d %d/%d\n", a, a, b, b, c, c);
                }
                fclose(file);

                // material
                sprintf(fullpath, "./obj_out/%s.mtl", filename);
                file = fopen(fullpath, "wb");
                if (file) {
                  fprintf(file, "# face mesh OBJ mtl exporter by andre.hl.chen@gmail.com 2022\n");
                  if (time) {
                    fprintf(file, "# File created %d/%02d/%02d %02d:%02d:%02d\n",
                            time->tm_year + 1900, time->tm_mon + 1,  time->tm_mday,
                            time->tm_hour, time->tm_min, time->tm_sec);
                  }
                  fprintf(file, "\nnewmtl face_from_camera\n");
                  fprintf(file, "  Ka 1.000 1.000 1.000\n"); // ambient
                  fprintf(file, "  Kd 1.000 1.000 1.000\n"); // diffuse
                  fprintf(file, "  Ks 0.000 0.000 0.000\n"); // specular
                  fprintf(file, "  Ke 0.000 0.000 0.000\n"); // emissive
                  fprintf(file, "  Tf 1.0000 1.0000 1.0000\n"); // Transmission Filter Color
                  fprintf(file, "  Ns 1.000\n"); // specular exponent
                  fprintf(file, "  d 1.000\n"); // opacity = 1.0, fully
                  fprintf(file, "  Tr 0.000\n"); // transparency = 1.0 - opacity
                  fprintf(file, "  Ni 1.500\n"); // optical density for its surface. i.e. index of refraction.
                  fprintf(file, "  illum 2\n"); // illumination model = 0 ~ 10
                  fprintf(file, "  map_Ka %s.jpg\n", filename); // ambient texture map
                  fprintf(file, "  map_Kd %s.jpg\n", filename); // diffuse texture map
                  fclose(file);

                  // texture file
                  sprintf(fullpath, "./obj_out/%s.jpg", filename);
                  cv::Mat img;
                  cvtColor(cv::Mat(render_task_.height, render_task_.width, CV_8UC3, (void*)render_task_.background), img, CV_RGB2BGR);
                  cv::imwrite(fullpath, img);
                }
              }

              trigger_save_OBJ_file_ = false;
            }

            // compare face_geometry::pose_transformation_matrix with our transform
            if (geometry_list) {
              auto const& ltm = geometry_list->pose_transform_matrix();
              assert(ltm.packed_data_size()==ltm.rows()*ltm.cols());
              auto err = xform;
              if (mediapipe::MatrixData_Layout_COLUMN_MAJOR==ltm.layout()) { // as default
                auto const* mtx = ltm.packed_data().data();
                err.m11 -= mtx[0]; err.m12 -= mtx[4]; err.m13 -= mtx[8];  err.m14 -= mtx[12];
                err.m21 -= mtx[1]; err.m22 -= mtx[5]; err.m23 -= mtx[9];  err.m24 -= mtx[13];
                err.m31 -= mtx[2]; err.m32 -= mtx[6]; err.m33 -= mtx[10]; err.m34 -= mtx[14];
              } else {
                auto const* mtx = ltm.packed_data().data();
                err.m11 -= mtx[0]; err.m12 -= mtx[1]; err.m13 -= mtx[2];  err.m14 -= mtx[3];
                err.m21 -= mtx[4]; err.m22 -= mtx[5]; err.m23 -= mtx[6];  err.m24 -= mtx[7];
                err.m31 -= mtx[8]; err.m32 -= mtx[9]; err.m33 -= mtx[10]; err.m34 -= mtx[11];
              }
              float translate_error = sqrt(err.m14*err.m14 + err.m24*err.m24 + err.m34*err.m34);
              float rot_error = fabs(err.m11) + fabs(err.m12) + fabs(err.m13) +  
                                fabs(err.m21) + fabs(err.m22) + fabs(err.m23) +  
                                fabs(err.m31) + fabs(err.m32) + fabs(err.m33);

              if ((translate_error+rot_error)>=print_pose_transform_error_threshold_) {
                printf("pose_transform_matrix error = {\n");
                printf("  % .6f % .6f % .6f  % .6f\n", err.m11, err.m12, err.m13, err.m14);
                printf("  % .6f % .6f % .6f  % .6f\n", err.m21, err.m22, err.m23, err.m24);
                printf("  % .6f % .6f % .6f  % .6f\n", err.m31, err.m32, err.m33, err.m34);
                printf("}\n");
                print_pose_transform_error_threshold_ = translate_error+rot_error;
              }

              ++geometry_list;
            }

            if (view_angle_3d_) {
              // xform = view * xform
              Matrix3 m = xform;
              xform.m11 = view.m11*m.m11 + view.m12*m.m21 + view.m13*m.m31;
              xform.m12 = view.m11*m.m12 + view.m12*m.m22 + view.m13*m.m32;
              xform.m13 = view.m11*m.m13 + view.m12*m.m23 + view.m13*m.m33;
              xform.m14 = view.m11*m.m14 + view.m12*m.m24 + view.m13*m.m34 + view.m14;

              xform.m21 = view.m21*m.m11 + view.m22*m.m21 + view.m23*m.m31;
              xform.m22 = view.m21*m.m12 + view.m22*m.m22 + view.m23*m.m32;
              xform.m23 = view.m21*m.m13 + view.m22*m.m23 + view.m23*m.m33;
              xform.m24 = view.m21*m.m14 + view.m22*m.m24 + view.m23*m.m34 + view.m24;

              xform.m31 = view.m31*m.m11 + view.m32*m.m21 + view.m33*m.m31;
              xform.m32 = view.m31*m.m12 + view.m32*m.m22 + view.m33*m.m32;
              xform.m33 = view.m31*m.m13 + view.m32*m.m23 + view.m33*m.m33;
              xform.m34 = view.m31*m.m14 + view.m32*m.m24 + view.m33*m.m34 + view.m34;
            }

            face_paint.Update(landmarks, 468);

            if (view_angle_3d_) {
              for (int i=0; i<468; ++i) {
                auto const& s = normalized_landmarks.landmark(i);
                auto& t = texcoords[i];
                t.x = s.x();
                t.y = s.y();
              }
              face_mesh_.Update(landmarks, texcoords, 468,
                                render_task_.background, render_task_.width, render_task_.height);

              face_mesh_.SetDepthWriteOffset(1); // put this somewhere for one time only
              face_mesh_.SetDepthWrite(true);
              OpenGLRenderer::Draw(view, face_mesh_);

              // line
              OpenGLRenderer::SetColor(0, 0, 255); // blue
              face_mesh_.SetMode(2);
              face_mesh_.SetTextureEnable(false);
              face_mesh_.SetDepthWrite(false);
              OpenGLRenderer::Draw(view, face_mesh_);
              face_mesh_.SetMode(0);
              face_mesh_.SetTextureEnable(true);
            } else {
              // draw occlusion
              face_paint.SetColorWrite(false);
              face_paint.SetDepthWriteOffset(1);
              OpenGLRenderer::Draw(view, face_paint);
              face_paint.SetColorWrite(true);
              face_paint.SetDepthWriteOffset(0);
            }

            // draw effect
            current_fx.SetMode(draw_mode_);
            if (&face_paint!=&current_fx) {
              // compansate scale difference between canonical face model and face mesh
              if (100!=canonical_face_model_scale_pct_) {
                float const ss = 0.01f*canonical_face_model_scale_pct_;
                float const inv_ss = 1.0f/ss;
                xform.m11 *= ss;
                xform.m12 *= ss;
                xform.m13 *= ss;
                xform.m14 *= inv_ss;

                xform.m21 *= ss;
                xform.m22 *= ss;
                xform.m23 *= ss;
                xform.m24 *= inv_ss;

                xform.m31 *= ss;
                xform.m32 *= ss;
                xform.m33 *= ss;
                xform.m34 *= inv_ss;
              }

              OpenGLRenderer::Draw(xform, current_fx);
            } else {
              OpenGLRenderer::Draw(view, face_paint);
            }
            current_fx.SetMode(0);

            // draw iris
            if (478==num_landmarks) {
              // back 5 mm, to make it visible
              auto* const iris = landmarks + 468;
              for (int i=0; i<10; ++i) {
                auto& v = iris[i];
                assert(v.z<0.0f);
                float zz = v.z + 0.5f;
                float ss = zz/v.z;
                v.x *= ss;
                v.y *= ss;
                v.z = zz;
              }
              OpenGLRenderer::SetColor(0, 255, 0); // color green
              OpenGLRenderer::Draw(view, iris, 10);
            }

            if (show_canonical_landmarks_) {
              OpenGLRenderer::SetDepthTestEnable(false);
              OpenGLRenderer::SetColor(255, 255, 0);
              if (1==show_canonical_landmarks_) {
                int totals = 0;
                for (int id:procrustes_landmark_indices_) {
                  landmarks[totals++] = canonical_face_model_[id];
                }
                OpenGLRenderer::Draw(xform, landmarks, totals, 8.0f);
              } else {
                face_paint.Update(canonical_face_model_.data(), (int)canonical_face_model_.size());
                face_paint.SetMode(2);
                face_paint.SetTextureEnable(false);
                OpenGLRenderer::Draw(xform, face_paint);
                face_paint.SetTextureEnable(true);
                face_paint.SetMode(0);
              }
              OpenGLRenderer::SetDepthTestEnable(true);
            }
          }
        }

        render_task_.landmark_set = nullptr;
      }
      render_task_.result = OpenGLRenderer::EndScene();
    }

    return true;
  }
  void OnDestroyGL_() override {
    face_mesh_.Release();
    for (int i=0; i<num_effect_meshes; ++i) {
      effect_meshes_[i].Release();
    }
  }
};

REGISTER_CALCULATOR(FaceEffectRenderer);

}  // namespace mediapipe
