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

#include "procrustes_solver.h"

#include <fstream>
#include <cassert>
#include <string>

#include <Eigen/SVD>

constexpr uint32_t geometry_file_magic = 0x12345678;

bool WeightedOrthogonalProblemSolver::ReadFromCache_(Mesh& mesh, std::vector<int>& procrustes_landmark_indices, char const* filename) {
  Reset_();
  procrustes_landmark_indices.clear();
  mesh.vertices.clear();
  mesh.indices.clear();

  FILE* file = fopen(filename, "rb");
  if (file) {
    uint32_t error = 0xffffffff;
    if ((4==fread(&error, 1, 4, file)) && geometry_file_magic==error) {
      int input_source(-1), vertex_type(0), primitive_type(0);
      int total_vertices(0), total_primitives(0);
      fread(&input_source, 1, 4, file);     // 1: FACE_LANDMARK_PIPELINE
      fread(&vertex_type, 1, 4, file);      // 5: XYZ+UV
      fread(&primitive_type, 1, 4, file);   // 3: triangle
      fread(&procrustes_landmark_basis_size_, 1, 4, file); // 33
      fread(&total_vertices, 1, 4, file);   // 468
      fread(&total_primitives, 1, 4, file); // 898

      if (procrustes_landmark_basis_size_>=3 && total_vertices>=468 && total_primitives>0 &&
          1==input_source && 5==vertex_type && 3==primitive_type) {

        int const require_data_size = procrustes_landmark_basis_size_*((int)sizeof(Vector3)*3 + 4);
        sqrt_weighted_targets_ = (Vector3*) malloc(require_data_size);
        if (sqrt_weighted_targets_) {
          sqrt_weighted_sources_ = sqrt_weighted_targets_ + procrustes_landmark_basis_size_;
          centered_weighted_sources_ = sqrt_weighted_sources_ + procrustes_landmark_basis_size_;
          
          int const procrustes_landmark_basis_data_size = procrustes_landmark_basis_size_*4;
          sqrt_weights_ = (float*) (centered_weighted_sources_ + procrustes_landmark_basis_size_);
          procrustes_landmark_indices.resize(procrustes_landmark_basis_size_);

          int const vertex_buffer_size = total_vertices*vertex_type*4;
          mesh.vertices.resize(total_vertices);
          
          int const index_buffer_size = total_primitives*3*4;
          mesh.indices.resize(total_primitives*3);

          if (procrustes_landmark_basis_data_size==fread(sqrt_weights_, 1, procrustes_landmark_basis_data_size, file) &&
              procrustes_landmark_basis_data_size==fread(procrustes_landmark_indices.data(), 1, procrustes_landmark_basis_data_size, file) &&
              vertex_buffer_size==fread(mesh.vertices.data(), 1, vertex_buffer_size, file) &&
              index_buffer_size==fread(mesh.indices.data(), 1, index_buffer_size, file)) {
            // pre-compute parameters
            Vector3 source_center_of_mass { 0.0f, 0.0f, 0.0f };
            for (int i=0; i<procrustes_landmark_basis_size_; ++i) {
              auto const& v = mesh.vertices[procrustes_landmark_indices[i]];
              float const w = sqrt_weights_[i]; // no sqrt
              float const w_sq = sqrt_weights_[i] = sqrt(w);
              Vector3& sws = sqrt_weighted_sources_[i];
              sws.x = w_sq*v.x;
              sws.y = w_sq*v.y;
              sws.z = w_sq*v.z;

              source_center_of_mass.x += w*v.x;
              source_center_of_mass.y += w*v.y;
              source_center_of_mass.z += w*v.z;
            }

            denominator_ = 0.0f;
            for (int i=0; i<procrustes_landmark_basis_size_; ++i) {
              auto const& sws = sqrt_weighted_sources_[i];
              float const w = sqrt_weights_[i];

              auto& cws = centered_weighted_sources_[i];
              cws.x = sws.x - w*source_center_of_mass.x;
              cws.y = sws.y - w*source_center_of_mass.y;
              cws.z = sws.z - w*source_center_of_mass.z;

              denominator_ += (cws.x*sws.x + cws.y*sws.y + cws.z*sws.z);
            }

            error = 0;
          } else {
            Reset_(); // failed
          }
        }
      }
    }
    fclose(file);
    return 0==error;
  }
  return false;
}

bool WeightedOrthogonalProblemSolver::ReadFromFile(Mesh& mesh, std::vector<int>& procrustes_landmark_indices, char const* filename) {
  assert(filename);

  // try load from binary cache file
  char cache_filename[256];
  {
    char const* pos = strrchr(filename, '/');
    if (!pos) {
      pos = strrchr(filename, '\\');
    }
    sprintf(cache_filename, "./%s.cache", pos ? (pos+1):filename);
  }
  if (ReadFromCache_(mesh, procrustes_landmark_indices, cache_filename)) {
    return true;
  }

  // OK... text file, great!
  // (mediapipe/modules/face_geometry/data/geometry_pipeline_metadata_landmarks.pbtxt)
  std::ifstream fin(filename);
  if (fin.is_open()) {
    double proc_weight_sum = 0.0;
    std::vector<int> proc_ids;
    std::vector<double> proc_weights;
    std::vector<float> vb;
    std::vector<int> ib;

    // reserve, all numbers come from line number of file geometry_pipeline_metadata_landmarks.pbtxt
    proc_ids.reserve(48-15);
    proc_weights.reserve(proc_ids.size());
    vb.reserve(2391-51);
    ib.reserve(5085-2391);

    int read_section = 0;
    int num_lines = 0;
    int input_source = -1;
    int vertex_type = 0; 
    int primitive_type = 0;
    int id;
    float x;
    double weight;
    bool loading_canonical_mesh = false;

    for (std::string line; std::getline(fin, line); ++num_lines) {
      auto const len = line.length();
      if (len>0) {
        char const* c_str = line.c_str();
        if ('#'==c_str[0]) {
          //printf("%s\n", c_str);
        } else if (0==read_section) {
          if (len>13 && 0==memcmp(c_str, "input_source:", 13)) {
            c_str += 13;
            while (' '==*c_str) {
              ++c_str;
            }

            if (0==memcmp(c_str, "FACE_LANDMARK_PIPELINE", 23)) {
              input_source = 1;
            } else if (0==memcmp(c_str, "FACE_DETECTION_PIPELINE", 24)) {
              input_source = 2;
            } else if (0==memcmp(c_str, "DEFAULT", 8)) {
              input_source = 0;
            }

            read_section = 1;
          } else if (len>12 && 0==memcmp(c_str, "vertex_type:", 12)) {
            c_str += 12;
            while (' '==*c_str) {
              ++c_str;
            }

            if (0==memcmp(c_str, "VERTEX_PT", 10)) {
              vertex_type = 5; // Position (XYZ) + Texture coordinate (UV)
              primitive_type = 0;
              read_section = 3; // skip to 3
              loading_canonical_mesh = false;
            } else {
              read_section = -7;
            }
          }
        } else if (1==read_section) {
          if (len>40 && 0==memcmp(c_str, "procrustes_landmark_basis { landmark_id:", 40)) {
            if ('}'==c_str[len-1] && 2==std::sscanf(c_str+40, "%d weight:%lf", &id, &weight)) {
              proc_ids.push_back(id);
              proc_weights.push_back(weight);
              proc_weight_sum += weight;
            } else {
              read_section = -read_section;
            }
          } else if (17==len && 0==memcmp(c_str, "canonical_mesh: {", 17)) {
            loading_canonical_mesh = true;
            vertex_type = primitive_type = 0;
            read_section = 2;
          } else {
            read_section = -read_section;
          }
        } else if (2==read_section) {
          assert(loading_canonical_mesh);
          if (len>14 && 0==memcmp(c_str, "  vertex_type:", 14)) {
            c_str += 14;
            while (' '==*c_str) {
              ++c_str;
            }

            if (0==memcmp(c_str, "VERTEX_PT", 10)) {
              vertex_type = 5; // Position (XYZ) + Texture coordinate (UV)
            }
          }

          if (vertex_type>0) {
            read_section = 3;
          } else {
            read_section = -read_section;
          }
        } else if (3==read_section) {
          if (loading_canonical_mesh && ' '==c_str[0] && ' '==c_str[1]) {
            c_str += 2;
          }

          if (len>17 && 0==memcmp(c_str, "primitive_type:", 15)) {
            c_str += 15;
            while (' '==*c_str) {
              ++c_str;
            }

            if (0==memcmp(c_str, "TRIANGLE", 9)) {
              primitive_type = 3;
            }
          }

          if (primitive_type>0) {
            read_section = 4;
          } else {
            read_section = -read_section;
          }
        } else if (4==read_section) {
          if (loading_canonical_mesh && ' '==c_str[0] && ' '==c_str[1]) {
            c_str += 2;
          }

          if (len>=15 && 0==memcmp(c_str, "vertex_buffer:", 14)) {
            if (1==std::sscanf(c_str+14, "%f", &x)) {
              vb.push_back(x);
            } else {
              read_section = -read_section;
            }
          } else if (len>=14 && 0==memcmp(c_str, "index_buffer:", 13)) {
            if (1==std::sscanf(c_str+13, "%d", &id)) {
              ib.push_back(id);
              read_section = 5;
            } else {
              read_section = -read_section;
            }
          } else {
            read_section = -read_section;
          }
        } else if (5==read_section) {
          if (loading_canonical_mesh && ' '==c_str[0] && ' '==c_str[1]) {
            c_str += 2;
          }
          if (len>=14 && 0==memcmp(c_str, "index_buffer:", 13)) {
            if (1==std::sscanf(c_str+13, "%d", &id)) {
              ib.push_back(id);
            } else {
              read_section = -read_section;
            }
          } else if ((1==len && '}'==c_str[0])) {
            read_section = 6;
          } else {
            read_section = -read_section;
          }
        }

        if (read_section<0) {
          printf("[ERROR line:%d section:%d] %s\n", num_lines+1, -read_section, c_str);
          break;
        }
      }
    }

    if (5==read_section && !loading_canonical_mesh) {
      read_section = 6;
    }

    int const proc_size = (int) proc_ids.size();
    int const vb_size = (int) vb.size();
    int const ib_size = (int) ib.size();
    if (6==read_section && vertex_type>=3 && primitive_type>0 && vb_size>0 && ib_size>0 &&
        0==(vb_size%vertex_type) && 0==(ib_size%primitive_type)) {
      int const total_vertices = vb_size/vertex_type;
      int const total_primitives = ib_size/primitive_type;

      //printf("vertices: %d primitives: %d procrustes: %d\n", total_vertices, total_primitives, proc_size);

      FILE* cache = fopen(cache_filename, "wb");
      if (cache) {
        fwrite(&geometry_file_magic, 1, 4, cache);
        fwrite(&input_source, 1, 4, cache);
        fwrite(&vertex_type, 1, 4, cache);
        fwrite(&primitive_type, 1, 4, cache);
        fwrite(&proc_size, 1, 4, cache);
        fwrite(&total_vertices, 1, 4, cache);
        fwrite(&total_primitives, 1, 4, cache);

        // procrustes landmark basis
        assert(proc_size==proc_weights.size());
        if (proc_size) {
          for (auto ww:proc_weights) {
            x = (float) (ww/proc_weight_sum);
            fwrite(&x, sizeof(x), 1, cache);
          }
          fwrite(proc_ids.data(), sizeof(proc_ids[0]), proc_size, cache);
        }

        // vertex buffer:
        // X:right, Y:up, Z: toward camera (as OpenGL coordinate system)
        fwrite(vb.data(), sizeof(vb[0]), vb_size, cache);

        // index buffer
        fwrite(ib.data(), sizeof(ib[0]), ib_size, cache);

        fclose(cache); cache = nullptr;
            
        return ReadFromCache_(mesh, procrustes_landmark_indices, cache_filename);
      }
    }
  }

  return false;
}

float WeightedOrthogonalProblemSolver::Solve(Matrix3* xform, Vector3 const* targets) const {
  float m11(0.0f), m12(0.0f), m13(0.0f);
  float m21(0.0f), m22(0.0f), m23(0.0f);
  float m31(0.0f), m32(0.0f), m33(0.0f);
  for (int i=0; i<procrustes_landmark_basis_size_; ++i) {
    auto const& t = targets[i];
    auto const sw = sqrt_weights_[i];
    auto& wt = sqrt_weighted_targets_[i];
    wt.x = sw * t.x;
    wt.y = sw * t.y;
    wt.z = sw * t.z;

    auto const& cws = centered_weighted_sources_[i];
    m11 += wt.x*cws.x;
    m12 += wt.x*cws.y;
    m13 += wt.x*cws.z;

    m21 += wt.y*cws.x;
    m22 += wt.y*cws.y;
    m23 += wt.y*cws.z;

    m31 += wt.z*cws.x;
    m32 += wt.z*cws.y;
    m33 += wt.z*cws.z;
  }

  {
    // design_matrix is a transposed LHS of (51) in the paper.
    Eigen::Matrix3f design_matrix(3, 3);
    design_matrix << m11, m12, m13, m21, m22, m23, m31, m32, m33;
    if (design_matrix.norm()<1.e-6f) {
      printf("Design matrix norm is too small!\n");
      return -1.0f;
    }
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(design_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Transposed (52) from the paper.
    //Eigen::Matrix3f postrotation = svd.matrixU();
    //Eigen::Matrix3f prerotation = svd.matrixV().transpose();
    //Eigen::Matrix3f rotation = postrotation * prerotation;
    Eigen::Matrix3f rotation = svd.matrixU() * (svd.matrixV().transpose());
    float const* data = rotation.data();  // column-major!
    m11 = data[0]; m12 = data[3]; m13 = data[6];
    m21 = data[1]; m22 = data[4]; m23 = data[7];
    m31 = data[2]; m32 = data[5]; m33 = data[8];
    assert(fabs(m11*m22*m33+m12*m23*m31+m21*m32*m13-m31*m22*m13-m11*m32*m23-m21*m12*m33-1.0f)<1.e-5f);
  }

  // tranposed(T) tranposed(A_w) (I - C).
  //const auto rotated_centered_weighted_sources = rotation * centered_weighted_sources;
  // Use the identity trace(A B) = sum(A * B^T)
  // to avoid building large intermediate matrices (* is Hadamard product).
  // (53) from the paper.
  //float numerator = rotated_centered_weighted_sources.cwiseProduct(weighted_targets).sum();
  float numerator = 0.0f;
  for (int i=0; i<procrustes_landmark_basis_size_; ++i) {
    Vector3 const& t = sqrt_weighted_targets_[i];
    Vector3 const& s = centered_weighted_sources_[i];
    numerator += t.x*(m11*s.x + m12*s.y + m13*s.z) +
                 t.y*(m21*s.x + m22*s.y + m23*s.z) +
                 t.z*(m31*s.x + m32*s.y + m33*s.z);
  }

  float const scale = numerator/denominator_;

  if (xform) {
    // scaling + rotation
    xform->m11 = m11 *= scale;
    xform->m12 = m12 *= scale;
    xform->m13 = m13 *= scale;
    xform->m21 = m21 *= scale;
    xform->m22 = m22 *= scale;
    xform->m23 = m23 *= scale;
    xform->m31 = m31 *= scale;
    xform->m32 = m32 *= scale;
    xform->m33 = m33 *= scale;

    //
    // Compute optimal translation for the weighted problem.
    //
    // tranposed(B_w - c A_w T) = tranposed(B_w) - R tranposed(A_w) in (54).
    //auto const pointwise_diffs = weighted_targets - rotation_and_scale * weighted_sources;

    // Multiplication by j_w is a respectively weighted column sum.
    // (54) from the paper.
    //auto const weighted_pointwise_diffs = pointwise_diffs.array().rowwise() * sqrt_weights.array().transpose();
    //Eigen::Vector3f translation = weighted_pointwise_diffs.rowwise().sum();

    // translation
    float& m14 = xform->m14 = 0.0f;
    float& m24 = xform->m24 = 0.0f;
    float& m34 = xform->m34 = 0.0f;
    for (int i=0; i<procrustes_landmark_basis_size_; ++i) {
      Vector3 const& t = sqrt_weighted_targets_[i];
      Vector3 const& s = sqrt_weighted_sources_[i];
      float const w = sqrt_weights_[i];

      // sum(sqrt_weights[i] * (weighted_target[i] - rotation_and_scale * weighted_sources[i]))
      m14 += w*(t.x - (m11*s.x + m12*s.y + m13*s.z));
      m24 += w*(t.y - (m21*s.x + m22*s.y + m23*s.z));
      m34 += w*(t.z - (m31*s.x + m32*s.y + m33*s.z));
    }
  }

  return scale;
}