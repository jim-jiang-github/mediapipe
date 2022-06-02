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

#ifndef FACE_EFFECT_MESH_H
#define FACE_EFFECT_MESH_H

#include <vector>

/*
 * transform Matrix3 apply on Vector3(x, y, z):
 *
 *  | x' | = | m11 m12 m13 m14 |   |  x  |
 *  | y' | = | m21 m22 m23 m24 | * |  y  |
 *  | z' | = | m31 m32 m33 m34 |   |  z  |
 *                                 | 1.0 |
*/
struct Matrix3 {
  float m11, m12, m13, m14;
  float m21, m22, m23, m24;
  float m31, m32, m33, m34;

  Matrix3(float x=0.0f, float y=0.0f, float z=0.0f):
    m11(1.0f),m12(0.0f),m13(0.0f),m14(x),
    m21(0.0f),m22(1.0f),m23(0.0f),m24(y),
    m31(0.0f),m32(0.0f),m33(1.0f),m34(z) {
  }
};

struct Vector3 {
  float x, y, z;
};

struct Vector2 {
  float x, y;
};

// mesh from file
struct Mesh {
  struct Vertex {
    float x, y, z;
    float u, v;
  };
  std::vector<Vertex> vertices;
  std::vector<int> indices; // triangle list
};

bool LoadMeshFrom_pbtxt(Mesh& mesh, char const* filename);

int GerenateLineListFromTriangleList(std::vector<int>& line_list,
                                     std::vector<int> const& triangle_list);

#endif
