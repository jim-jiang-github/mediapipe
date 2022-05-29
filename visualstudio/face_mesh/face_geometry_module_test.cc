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
// to test FaceGeometryPipelineCalculator module, insert following to your main graph...
//
// node {
//   calculator: "ImagePropertiesCalculator"
//   input_stream: "IMAGE:throttled_input_video"
//   output_stream: "SIZE:image_size"
// }
//
// node {
//   calculator: "FaceGeometryEnvGeneratorCalculator"
//   output_side_packet: "ENVIRONMENT:environment"
//   node_options: {
//     [type.googleapis.com/mediapipe.FaceGeometryEnvGeneratorCalculatorOptions] {
//       environment: {
//         origin_point_location: TOP_LEFT_CORNER
//         perspective_camera: {
//           vertical_fov_degrees: 63  # SR300:42.7
//           near: 1.0  # 1cm
//           far: 10000.0  # 100m
//         }
//       }
//     }
//   }
// }
//
// node {
//   calculator: "FaceGeometryPipelineCalculator"
//   input_side_packet: "ENVIRONMENT:environment"
//   input_stream: "IMAGE_SIZE:image_size"
//   input_stream: "MULTI_FACE_LANDMARKS:multi_face_landmarks"
//   output_stream: "MULTI_FACE_GEOMETRY:multi_face_geometry"
//   options: {
//     [mediapipe.FaceGeometryPipelineCalculatorOptions.ext] {
//       metadata_path: "../../mediapipe/modules/face_geometry/data/geometry_pipeline_metadata_landmarks.pbtxt"
//     }
//   }
// }
//
// node {
//   calculator: "FaceGeometryModuleCheckCalculator"
//   input_side_packet: "ENVIRONMENT:environment"
//   input_stream: "USER_INPUT:user_input"
//   input_stream: "IMAGE_SIZE:image_size"
//   input_stream: "MULTI_FACE_LANDMARKS:multi_face_landmarks"
//   input_stream: "MULTI_FACE_GEOMETRY:multi_face_geometry"
//   output_stream: "DUMMY_OUTPUT:dummy_output"
//
//   options: {
//     [mediapipe.FaceGeometryPipelineCalculatorOptions.ext] {
//       metadata_path: "../../mediapipe/modules/face_geometry/data/geometry_pipeline_metadata_landmarks.pbtxt"
//     }
//   }
// }
//

#include "../calculator_graph_util.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/modules/face_geometry/protos/environment.pb.h"
#include "mediapipe/modules/face_geometry/protos/face_geometry.pb.h"
#include "mediapipe/modules/face_geometry/geometry_pipeline_calculator.pb.h"
#include "mediapipe/modules/face_geometry/protos/geometry_pipeline_metadata.pb.h"
#include "mediapipe/util/resource_util.h"

namespace mediapipe {

constexpr char const* kUserInputTag = "USER_INPUT";
constexpr char const* kEnvironmentTag = "ENVIRONMENT";
constexpr char const* kImageSizeTag = "IMAGE_SIZE";
constexpr char const* kMultiFaceGeometryTag = "MULTI_FACE_GEOMETRY";
constexpr char const* kMultiFaceLandmarksTag = "MULTI_FACE_LANDMARKS";
constexpr char const* kDummyOutputTag = "DUMMY_OUTPUT";

class FaceGeometryModuleCheckCalculator : public CalculatorBase {
  struct procrustes_landmark_basis {
    float x, y, z;
    float weight;
    int id;
  };
  std::vector<procrustes_landmark_basis> procrustes_landmark_basis_;

  float vertical_fov_degrees_ { 43.0f };
  bool on_idle_ {};

public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets().Tag(kEnvironmentTag).Set<face_geometry::Environment>();

    auto& inputs = cc->Inputs();
    inputs.Tag(kUserInputTag).Set<UserInput>();
    inputs.Tag(kImageSizeTag).Set<std::pair<int, int>>();
    inputs.Tag(kMultiFaceLandmarksTag).Set<std::vector<NormalizedLandmarkList>>();
    inputs.Tag(kMultiFaceGeometryTag).Set<std::vector<face_geometry::FaceGeometry>>();
    cc->Outputs().Tag(kDummyOutputTag).Set<int>();

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(mediapipe::TimestampDiff(0));
    {
      face_geometry::GeometryPipelineMetadata metadata;
      auto const& metadata_path = cc->Options<FaceGeometryPipelineCalculatorOptions>().metadata_path();
      std::string metadata_blob; // 140 KB
      MP_RETURN_IF_ERROR(mediapipe::GetResourceContents(metadata_path, &metadata_blob)) << "Failed to read content blob! Resolved path = " << metadata_path;
      RET_CHECK(google::protobuf::TextFormat::ParseFromString(metadata_blob, &metadata)) << "Failed to parse a metadata proto from a text blob!";

      auto const& procrustes_landmark_basis = metadata.procrustes_landmark_basis();
      float const* const canonical_mesh_vb = metadata.canonical_mesh().vertex_buffer().data();
      procrustes_landmark_basis_.clear();
      procrustes_landmark_basis_.reserve(procrustes_landmark_basis.size());
      for (auto const& wi : procrustes_landmark_basis) {
        int const id = (int) wi.landmark_id();
        auto const ww = wi.weight();
        float const* v = canonical_mesh_vb + 5*id;
        procrustes_landmark_basis_.push_back({v[0], v[1], v[2], ww, id});
      }
    }

    vertical_fov_degrees_ =
        cc->InputSidePackets()
            .Tag(kEnvironmentTag)
            .Get<face_geometry::Environment>().perspective_camera().vertical_fov_degrees();

    on_idle_ = false;

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    auto const& inputs = cc->Inputs();

    if (on_idle_) {
      // push SPACEBAR to do again
      if (inputs.HasTag(kUserInputTag) && !inputs.Tag(kUserInputTag).IsEmpty()) {
        auto const& input = inputs.Tag(kUserInputTag).Get<UserInput>();
        if (' '==input.wait_key) {
          on_idle_ = false;
        }
      }
    }

    int result = 0;
    if (!on_idle_ && !inputs.Tag(kMultiFaceLandmarksTag).IsEmpty() && !inputs.Tag(kMultiFaceGeometryTag).IsEmpty()) {
      auto const& landmarks_list = inputs.Tag(kMultiFaceLandmarksTag).Get<std::vector<NormalizedLandmarkList>>();
      auto const& geometry_list = inputs.Tag(kMultiFaceGeometryTag).Get<std::vector<face_geometry::FaceGeometry>>();
      int const num_faces = (int) landmarks_list.size();
      if (num_faces==geometry_list.size()) {
        for (int i=0; i<num_faces; ++i) {
          face_geometry::FaceGeometry const& face_geometry = geometry_list[i];
          if (!face_geometry.has_pose_transform_matrix()) {
            result = -3;
            break;
          }

          //
          // test #1 pose_transform_matrix
          auto const& ltm = face_geometry.pose_transform_matrix();
          assert(ltm.packed_data_size()==ltm.rows()*ltm.cols());
          float m11, m12, m13, m14;
          float m21, m22, m23, m24;
          float m31, m32, m33, m34;
          float m41, m42, m43, m44;
          if (mediapipe::MatrixData_Layout_COLUMN_MAJOR==ltm.layout()) { // as default
            auto const* mtx = ltm.packed_data().data();
            m11 = mtx[0]; m12 = mtx[4]; m13 = mtx[8];  m14 = mtx[12];
            m21 = mtx[1]; m22 = mtx[5]; m23 = mtx[9];  m24 = mtx[13];
            m31 = mtx[2]; m32 = mtx[6]; m33 = mtx[10]; m34 = mtx[14];
            m41 = mtx[3]; m42 = mtx[7]; m43 = mtx[11]; m44 = mtx[15];
          } else {
            auto const* mtx = ltm.packed_data().data();
            m11 = mtx[0]; m12 = mtx[1]; m13 = mtx[2];  m14 = mtx[3];
            m21 = mtx[4]; m22 = mtx[5]; m23 = mtx[6];  m24 = mtx[7];
            m31 = mtx[8]; m32 = mtx[9]; m33 = mtx[10]; m34 = mtx[11];
            m41 = mtx[12]; m42 = mtx[13]; m43 = mtx[14]; m44 = mtx[15];
          }
          printf("pose_transform_matrix[%d][%d] = {\n", ltm.rows(), ltm.cols());
          printf("  % .6f % .6f % .6f  %.6f\n", m11, m12, m13, m14);
          printf("  % .6f % .6f % .6f  %.6f\n", m21, m22, m23, m24);
          printf("  % .6f % .6f % .6f  %.6f\n", m31, m32, m33, m34);
          printf("  % .6f % .6f % .6f  %.6f\n", m41, m42, m43, m44);
          printf("}\n");

          //
          // test #2 mesh
          auto const& mesh = face_geometry.mesh();
          int const vertex_buffer_size = mesh.vertex_buffer_size();
          int const index_buffer_size = mesh.index_buffer_size();
          if (0!=(vertex_buffer_size%5) || 0!=(index_buffer_size%3)) {
            result = -4;
            break;
          }
          auto const& norm_landmarks = landmarks_list[i];
          int const& num_landmarks = norm_landmarks.landmark_size();
          int const num_vertices = vertex_buffer_size/5; // each vertexs consists of 5 floats: XYZ + UV
          int const num_triangles = index_buffer_size/3;
          printf("\n#(landmarks)=%d  #(vertices)=%d  #(triangles)=%d\n", num_landmarks, num_vertices, num_triangles);
          if (468!=num_vertices || (468!=num_landmarks && 478!=num_landmarks)) {
            result = -5;
            break;
          }

          //
          // test #3 check few points:
          //          #4 = nosetip
          //        #133 = right eye inside corner.
          //        #362 = left eye inside corner.
          printf("\ncheck points:\n");
          int const check_points[] = { 4, 133, 362 };
          float const* const realtime_model_vb = mesh.vertex_buffer().data();
          for (int k=0; k<(int)(sizeof(check_points)/sizeof(check_points[0])); ++k) {
            int const id = check_points[k];
            auto const& m = norm_landmarks.landmark(id);
            float const* p = realtime_model_vb + 5*id;
            printf("  #%03d (%f, %f, %f) -> (%f, %f, %f)\n", id, m.x(), m.y(), m.z(), p[0], p[1], p[2]);
          }
#if 0
          //
          // /mediapipe/modules/face_geometry/libs/geometry_pipeline.cc line.205

          //
          // Multiply each of the metric landmarks by the inverse pose
          // transformation matrix to align the runtime metric face landmarks with
          // the canonical metric face landmarks.
          metric_landmarks = (pose_transform_mat.inverse() *
                              metric_landmarks.colwise().homogeneous())
                                .topRows(3);
#endif
          //
          // test #4 procrustes analysis result. how fit is the face_mesh with canonical face model
          //         since you're not exactly the same shape as canonical face model. don't expect
          //         zero error.
          float avg_ex(0.0f), avg_ey(0.0f), avg_ez(0.0f), ex, ey, ez;
          printf("\nprocrustes analysis result:\n    #  canonical model     runtime landmarks\n");
          for (auto const& pp : procrustes_landmark_basis_) {
            float const* p = realtime_model_vb + 5*pp.id;
            avg_ex += ex = pp.x - p[0];
            avg_ey += ey = pp.y - p[1];
            avg_ez += ez = pp.z - p[2];
            printf("  %03d (% .1f,% .1f,% .1f) vs. (% .1f,% .1f,% .1f)  e:(%+.2f, %+.2f, %+.2f)\n",
                   pp.id, pp.x, pp.y, pp.z, p[0], p[1], p[2], ex, ey, ez);
          }
          float const plb_size = (float) procrustes_landmark_basis_.size();
          printf(" => avg error: z=%+.2f y=%+.2f z=%+.2f cm\n", avg_ex/=plb_size, avg_ey/=plb_size, avg_ez/=plb_size);

          //
          // test #5 repojection error. should be zero or the algorithm changes landmarks pixel coordinate...
          auto const& image_size = inputs.Tag(kImageSizeTag).Get<std::pair<int, int>>();
          float const img_width  = (float) image_size.first;
          float const img_height = (float) image_size.second;
          float const pixel_cx = 0.5f*(img_width-0.0f);
          float const pixel_cy = 0.5f*(img_height-0.0f);
          float const pixel_scale = pixel_cy/tan(0.5f*vertical_fov_degrees_*0.0174533f);
          float u0, v0, u1, v1, x, y, z;
          avg_ex = avg_ey = 0.0f;
          printf("\ncheck re-projection error: (vertical fov=%.1f degree)\n", vertical_fov_degrees_);
          for (int i=0; i<num_vertices; ++i) {
            auto const& m = norm_landmarks.landmark(i);
            u0 = m.x()*img_width;
            v0 = m.y()*img_height;

            float const* p = realtime_model_vb + 5*i;
            x = p[0]; y = p[1]; z = p[2];

            // OpenGL coordinate system
            ex = m11*x + m12*y + m13*z + m14;
            ey = m21*x + m22*y + m23*z + m24;
            ez = m31*x + m32*y + m33*z + m34;
            assert(ez<0.0f); // Z: toward camera

            // project 
            u1 = pixel_cx - pixel_scale*ex/ez; // X: right
            v1 = pixel_cy + pixel_scale*ey/ez; // Y: up

            ex = u1 - u0;
            ey = v1 - v0;
            avg_ex += fabs(ex);
            avg_ey += fabs(ey);
            if (fabs(ex)>0.1f || fabs(ey)>0.1f) {
              printf("  %03d (%.1f, %.1f) vs. (%.1f, %.1f)  e:(%+.2f, %+.2f) pixels\n",
                     i, u0, v0, u1, v1, ex, ey);
            }
          }

          avg_ex /= num_vertices; avg_ey /= num_vertices;
          if (avg_ex<0.01f && avg_ey<0.01f) {
            printf(" => perfect! No error!\n\n");
          } else {
            printf(" => avg abs_err: u=%+.2f v=%+.2f pixels\n\n", avg_ex, avg_ey);
          }

          printf("*** press SPACEBAR to test again ***\n\n");
          on_idle_ = true;

          ++result;
        }
      } else {
        result = -2;
      }
    }

    //cc->Outputs().Tag(kDummyOutputTag).AddPacket(mediapipe::MakePacket<int>(result).At(cc->InputTimestamp()));
    return absl::OkStatus();
  }
};

REGISTER_CALCULATOR(FaceGeometryModuleCheckCalculator);

}