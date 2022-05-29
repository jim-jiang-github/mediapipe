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

#include "mesh.h"

#include <fstream>
#include <string>

namespace {

uint32_t const mesh_cache_file_magic = 0x89ABCDEF;
bool read_from_cache(Mesh& mesh, char const* filename) {
  FILE* file = fopen(filename, "rb");
  if (file) {
    uint32_t magic = 0;
    int vertex_type(0), num_vertices(0), num_indices(0);
    if (4==fread(&magic, 1, 4, file) && mesh_cache_file_magic==magic) {
      fread(&vertex_type, 1, 4, file);
      fread(&num_vertices, 1, 4, file);
      fread(&num_indices, 1, 4, file);
      if (5==vertex_type && num_vertices>2 && num_indices>2 && 0==(num_indices%3)) {
        mesh.vertices.resize(num_vertices);
        mesh.indices.resize(num_indices);
        fread(mesh.vertices.data(), sizeof(Mesh::Vertex), num_vertices, file);
        fread(mesh.indices.data(), sizeof(int), num_indices, file);
      }
    }
    fclose(file);

    return !mesh.vertices.empty() && !mesh.indices.empty();
  }

  return false;
}

}

bool LoadMeshFrom_pbtxt(Mesh& mesh, char const* filename) {
  mesh.vertices.clear();
  mesh.indices.clear();

  // try load from binary cache file
  char cache_filename[256];
  {
    char const* pos = strrchr(filename, '/');
    if (!pos) {
      pos = strrchr(filename, '\\');
    }
    sprintf(cache_filename, "./%s.cache", pos ? (pos+1):filename);
    if (read_from_cache(mesh, cache_filename)) {
      return true;
    }
  }

  std::ifstream fin(filename);
  if (fin.is_open()) {
    Mesh::Vertex vtx;
    float* v_ptr = (float*) &vtx;
    int v_offset = 0;

/*
    vertex_type: VERTEX_PT
    primitive_type: TRIANGLE
    vertex_buffer: -0.100000
    vertex_buffer: -0.100000
      .
      .
      .
    index_buffer: 0
    index_buffer: 1
    index_buffer: 2
      .
      .
      .
*/
    int num_lines(0), read_section(0), vertex_type(0), primitive_type(0), id;
    for (std::string line; std::getline(fin, line); ++num_lines) {
      auto const len = line.length();
      if (len>0) {
        char const* c_str = line.c_str();
        if ('#'==c_str[0]) {
          // printf("%s\n", c_str);
        } else if (0==read_section) {
          if (len>12 && 0==memcmp(c_str, "vertex_type:", 12)) {
            c_str += 12;
            while (' '==*c_str) {
              ++c_str;
            }

            if (0==memcmp(c_str, "VERTEX_PT", 10)) {
              vertex_type = 5; // Position (XYZ) + Texture coordinate (UV)
              primitive_type = 0;
              read_section = 1;
            } else {
              read_section = -7;
            }
          }
        } else if (1==read_section) {
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
            read_section = 2;
          } else {
            read_section = -read_section;
          }
        } else if (2==read_section) {
          if (len>=15 && 0==memcmp(c_str, "vertex_buffer:", 14)) {
            if (1==std::sscanf(c_str+14, "%f", v_ptr+v_offset)) {
              if (++v_offset>=5) {
                mesh.vertices.push_back(vtx);
                v_offset = 0;
              }
            } else {
              read_section = -read_section;
            }
          } else if (len>=14 && 0==memcmp(c_str, "index_buffer:", 13)) {
            if (1==std::sscanf(c_str+13, "%d", &id)) {
              mesh.indices.push_back(id);
              read_section = 3;
            } else {
              read_section = -read_section;
            }
          } else {
            read_section = -read_section;
          }
        } else if (3==read_section) {
          if (len>=14 && 0==memcmp(c_str, "index_buffer:", 13)) {
            if (1==std::sscanf(c_str+13, "%d", &id)) {
              mesh.indices.push_back(id);
            } else {
              read_section = -read_section;
            }
          } else {
            read_section = -read_section;
          }
        }
      }
    }

    int const num_vertices = (int) mesh.vertices.size();
    int const num_indices = (int) mesh.indices.size();
    //printf("%s vertices:%d indices:%d\n", filename, num_vertices, num_indices);

    if (5==vertex_type && 3==primitive_type && num_vertices>2 &&
      num_indices>2 && 0==(num_indices%3)) {
      // save cached file
      FILE* file = fopen(cache_filename, "wb");
      if (file) {
        fwrite(&mesh_cache_file_magic, 1, 4, file);
        fwrite(&vertex_type, 1, 4, file);
        fwrite(&num_vertices, 1, 4, file);
        fwrite(&num_indices, 1, 4, file);
        fwrite(mesh.vertices.data(), sizeof(Mesh::Vertex), num_vertices, file);
        fwrite(mesh.indices.data(), sizeof(int), num_indices, file);
        fclose(file);
      }

      return true;
    }
  }
  return false;
}

int GerenateLineListFromTriangleList(std::vector<int>& line_list,
                                     std::vector<int> const& triangle_list) {
  int const total_indices = (int) triangle_list.size();
  struct {
    int a, b;
  } line;
  std::vector<decltype(line)> lines;
  lines.reserve(total_indices);

  int const* indices = triangle_list.data();
  for (int i=0; i<total_indices; i+=3) {
    int i0 = indices[2], i1;
    for (int j=0; j<3; ++j,i0=i1) {
      i1 = *indices++;
      //assert(i1!=i0);
      if (i0<i1) {
        line.a = i0;
        line.b = i1;
      } else {
        line.a = i1;
        line.b = i0;
      }

      if (lines.end()==std::find_if(lines.begin(), lines.end(),
                                    [line](auto const& e) {
                                      return e.a==line.a && e.b==line.b;
                                    })) {
        lines.push_back(line);
      }
    }
  }

  int const total_lines = (int) lines.size();
  line_list.reserve(total_lines*2);
  for (auto const& l : lines) {
    line_list.push_back(l.a);
    line_list.push_back(l.b);
  }
  return total_lines;
}
