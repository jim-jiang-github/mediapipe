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

namespace {
constexpr char const* kImageTag = "IMAGE";
constexpr char const* kNormLandmarksTag = "NORM_LANDMARKS";
constexpr char const* kUserInputTag = "USER_INPUT";

constexpr int num_effect_meshes = 3;

void print_help() {
  printf("\n[INFO]\nface_effect key controls:\n");
  printf("      Effect: '1', '2', '3' or SPACEBAR\n");
  printf("   Primitive: 'm' for triangles, lines or points\n");
  printf("  Camera FOV: 'a' or 's'\n");
  printf(" Model Scale: 'z' or 'x'\n");
  printf("        HELP: 'h'\n\n");
}

} // namespace

namespace mediapipe {

class FaceEffectRenderer : public CalculatorBase, private OpenGLRenderer {
  // to solve landmarks from image space to metric space
  WeightedOrthogonalProblemSolver wop_solver_;
  std::vector<int> procrustes_landmark_indices_;

  // effect to show
  DrawMesh effect_meshes_[num_effect_meshes];

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

  // tweakable...
  float camera_vertical_fov_{43.0f};
  uint8_t draw_mode_{0};
  uint8_t current_effect_{0};
  uint8_t canonical_face_model_scale_pct_{100};
  uint8_t canonical_face_model_anchor_{100}; // ???

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

  static absl::Status GetContract(CalculatorContract* cc) {
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
      }
    }

    if (31==input_checks) {
      return absl::OkStatus();
    }

    return absl::InternalError("FaceEffectRenderer::GetContract() - illegal parameters");
  }

  absl::Status Open(CalculatorContext* cc) override {
    if (!cc->Inputs().HasTag(kImageTag)) {
      return absl::InternalError("FaceEffectRenderer::Open() - no image frame input not available");
    }
    calc_ctx_ = cc; // OnInitedGL_()
    OpenGLRenderer::Run_();

    cc->SetOffset(TimestampDiff(0));

    print_help();

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    auto const& inputs = cc->Inputs();

    // user input
    if (inputs.HasTag(kUserInputTag) && !inputs.Tag(kUserInputTag).IsEmpty()) {
      auto const& input = inputs.Tag(kUserInputTag).Get<UserInput>();

      // key
      if (' '==input.wait_key) {
        current_effect_ = (current_effect_+1)%3;
        printf("[INFO] Effect= %d\n", (int) current_effect_);
      } else if ('1'<=input.wait_key && input.wait_key<='1'+num_effect_meshes) {
        current_effect_ = input.wait_key - '1';
        printf("[INFO] Effect= %d\n", (int) current_effect_);
      } else if ('m'==input.wait_key) {
        //draw_mode_ = (2==draw_mode_) ? 0:2; // 0:triangles, 2:lines
        draw_mode_ = (draw_mode_+1)%3;
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
      } else if ('h'==input.wait_key) {
        print_help();
      } else if (input.wait_key>0) {
        printf("[INFO] key %08X not handled\n", input.wait_key);
      }

      // mouse
      if (cv::EVENT_LBUTTONDOWN==input.mouse.event) {
        if (cv::EVENT_FLAG_LBUTTON&input.mouse.flags) {
          printf("[INFO] LBUTTON down at: %d, %d\n", input.mouse.x, input.mouse.y);
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

      // kickout rendering
      if (render_task_.background) {
        OpenGLRenderer::FrameMove_();
      }

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
        auto output_frame = absl::make_unique<ImageFrame>(target_format, render_task_.width, render_task_.height);
        output_frame->CopyPixelData(target_format, render_task_.width, render_task_.height, render_frame.data,
                                    ImageFrame::kDefaultAlignmentBoundary);
        outputs.Tag(kImageTag).Add(output_frame.release(), cc->InputTimestamp());
      }
    }

    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    render_task_.Reset();
    OpenGLRenderer::Stop_();
    return absl::OkStatus();
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

      //
      // [0] : glasses.pbtxt + glasses.pngblob
      // [1] : axis.pbtxt + axis.pngblob
      // [2] : canonical face mesh + facepaint.pngblob
      char filename[256];
      sprintf(filename, "%s/facepaint.pngblob", model_path);
      if (effect_meshes_[2].Create(mesh, filename)) {
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
        OpenGLRenderer::BeginScene(render_task_.background, render_task_.width, render_task_.height)) {
      // draw each landmarks
      if (render_task_.landmark_set) {
        Matrix3 const identity;
        Matrix3 xform;      // transform canonical face model to landmarks in metric 3D space
        Vector3 landmarks[478]; // landmarks in metric 3D space

        auto& face_mesh = effect_meshes_[2];
        auto& current_fx = effect_meshes_[current_effect_%num_effect_meshes];
        for (auto const& normalized_landmarks: *render_task_.landmark_set) {
          int const num_landmarks = normalized_landmarks.landmark_size();
          if (SolveFaceLandmark3D_(xform, landmarks, normalized_landmarks)) {
            face_mesh.Update(landmarks, 468);
            // draw occlusion
            face_mesh.SetColorWrite(false);
            face_mesh.SetDepthOffset(1);
            OpenGLRenderer::Draw(identity, face_mesh);
            face_mesh.SetColorWrite(true);
            face_mesh.SetDepthOffset(0);

            // draw effect
            current_fx.SetMode(draw_mode_);
            if (&face_mesh!=&current_fx) {
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
              OpenGLRenderer::Draw(identity, face_mesh);
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
              OpenGLRenderer::Draw(identity, iris, 10);
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
    for (int i=0; i<num_effect_meshes; ++i) {
      effect_meshes_[i].Release();
    }
  }
};

REGISTER_CALCULATOR(FaceEffectRenderer);

}  // namespace mediapipe
