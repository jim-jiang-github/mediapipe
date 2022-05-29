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

#ifndef FACE_EFFECT_PROCRUSTE_SOLVER_H
#define FACE_EFFECT_PROCRUSTE_SOLVER_H

#include "mesh.h"

//
// The weighted problem is thoroughly addressed in Section 2.4 of:
// D. Akca, Generalized Procrustes analysis and its applications
// in photogrammetry, 2003, https://doi.org/10.3929/ethz-a-004656648
//
// Here, WeightedOrthogonalProblemSolver is a simplfied (and fast) implmention of
// mediapipe/mediapipe/modules/face_geometry/libs/procrustes_solver.cc
//
class WeightedOrthogonalProblemSolver {
  // pre-compute coeffieients
  Vector3* sqrt_weighted_targets_ {nullptr}; // working buffer
  Vector3* sqrt_weighted_sources_ {nullptr};
  Vector3* centered_weighted_sources_ {nullptr};
  float* sqrt_weights_ {nullptr};

  float denominator_ {0.0f};
  int procrustes_landmark_basis_size_ {0};

  bool ReadFromCache_(Mesh& mesh, std::vector<int>& procrustes_landmarkindices, char const* filename);

  void Reset_() {
    if (sqrt_weighted_targets_) {
      free(sqrt_weighted_targets_);
      sqrt_weighted_targets_ = nullptr;
    }
    sqrt_weighted_sources_ = nullptr;
    centered_weighted_sources_ = nullptr;
    sqrt_weights_  = nullptr;
  }

public:
  WeightedOrthogonalProblemSolver() = default;
  ~WeightedOrthogonalProblemSolver() { Reset_(); }

  // read from file with byproduct mesh
  bool ReadFromFile(Mesh& mesh, std::vector<int>& procrustes_landmark_indices, char const* filename);

  // solve transform matrix, return scale
  float Solve(Matrix3* xform, Vector3 const* targets) const;
};

#endif
