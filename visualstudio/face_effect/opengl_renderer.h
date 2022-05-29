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

#ifndef FACE_EFFECT_OPENGL_RENDERER_H
#define FACE_EFFECT_OPENGL_RENDERER_H

#include "mesh.h"

#include <mutex>
#include <condition_variable>

//
// Note: GLuint, GLint should be used instead of uint32_t,
// but i want to keep it simple and clean. (for the building system)
//
class DrawMesh final {
  uint32_t bos_[2]{0,0};  // buffer object, vertex and index 
  uint32_t texture_{0};

  int num_vertices_{0};
  int num_triangles_{0};
  int num_lines_{0};

  // render states
  uint8_t mode_:2; // 1: point, 2:line, 0/3:triangle
  uint8_t transparent_:1;
  uint8_t color_disable_:1;
  uint8_t depth_disable_:1;
  uint8_t zpass_:1;
  uint8_t cull_:2;
  int8_t  depth_offset_{0};

public:
  DrawMesh() = default;
  DrawMesh(DrawMesh const&) = delete;
  DrawMesh& operator=(DrawMesh const&) = delete;
  ~DrawMesh() { Release(); }

  void SetColorWrite(bool enable) { color_disable_ = !enable; }
  void SetDepthWrite(bool enable) { depth_disable_ = !enable; }
  void SetDepthOffset(int8_t offset) { depth_offset_ = offset; }
  void SetMode(uint8_t mode) { if (mode<5) mode_ = mode; }

  bool Create(Mesh const& mesh, char const* texture_name);
  bool Update(Vector3 const* xyz, int size);

  // release OpenGL release
  void Release();

  // vip
  friend class OpenGLRenderer;
};

class OpenGLRenderer {
  std::thread render_thread_;
  std::mutex render_wait_mutex_;
  std::condition_variable render_wait_cond_var_;
  std::mutex render_done_mutex_;
  std::condition_variable render_done_cond_var_;

  struct {
    // vertical_fov: camera field of view in radian
    // 0.0 < near_plane < far_plane
    float vertical_fov{1.0f}, near_plane{1.0f}, far_plane{100.0f};

    // other elements of matrix 4x4 are zero, except m43 = -1.0, 
    float m11, m22, m33, m34;

    // concatenate model_view to get model-view-projection matrix 4x4
    void MVP(float m[16], Matrix3 const& mv) const {
      // row-major
      m[0] = m11*mv.m11;
      m[1] = m11*mv.m12;
      m[2] = m11*mv.m13;
      m[3] = m11*mv.m14;

      m[4] = m22*mv.m21;
      m[5] = m22*mv.m22;
      m[6] = m22*mv.m23;
      m[7] = m22*mv.m24;

      m[8] = m33*mv.m31;
      m[9] = m33*mv.m32;
      m[10] = m33*mv.m33;
      m[11] = m33*mv.m34 + m34;

      m[12] = -mv.m31;
      m[13] = -mv.m32;
      m[14] = -mv.m33;
      m[15] = -mv.m34;
    }
  } perspective_projection_;

  // default color
  float color_[4]{1.0f, 1.0f, 1.0f, 1.0f};

  uint8_t* read_pixels_;

  uint32_t renderbuffers_[2]; // COLOR + DEPTH
  uint32_t framebuffer_;

  uint32_t fullscren_shader_;
  uint32_t fullscreen_texture_;

  uint32_t textured_model_shader_;
  uint32_t color_model_shader_;

  int textured_model_shader_mvp_loc_;
  int color_model_shader_mvp_loc_;
  int color_model_shader_color_loc_;

  int width_, height_;

  volatile int run_status_ {};

  // [opengl thread] i'll call you back.
  virtual bool OnInitedGL_() = 0;
  virtual bool DrawFrame_() = 0;
  virtual void OnDestroyGL_() = 0;

  bool InitGLObjects_();
  void DestroyGLObjects_();

protected:
  OpenGLRenderer();
  virtual ~OpenGLRenderer() {}

  // run & stop opengl thread
  bool Run_();
  void Stop_()  {
    if (render_thread_.joinable()) {
      run_status_ = -1;
      render_done_cond_var_.notify_one();
      render_wait_cond_var_.notify_one();
      render_thread_.join();
    }
  }

  // ready to draw a frame
  void FrameMove_() {
    if (0==run_status_) {
      run_status_ = 1;
      render_wait_cond_var_.notify_one();
      std::unique_lock<std::mutex> lock(render_done_mutex_);
      while (1==run_status_) {
        render_done_cond_var_.wait(lock);
      }
    }
  }

public:
  // render begins with fullscreen background
  // [width, height] size of background, also the frame buffer/render buffer size
  // [vfov] perspective camera's vertical field of view in degree
  bool BeginScene(uint8_t const* background, int width, int height);

  // set model's base color(when no texture apply on), wireframe color
  void SetColor(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha=255) {
    color_[0] = red/255.0f;
    color_[1] = green/255.0f;
    color_[2] = blue/255.0f;
    color_[3] = alpha/255.0f;
  }

  bool SetCameraPerspective(float vfov, float near_plane=1.0f, float far_plane=100.0f) {
    if (0.0f<vfov && vfov<180.0f && 0.0f<near_plane && near_plane<far_plane) {
      perspective_projection_.vertical_fov = vfov*0.01745329252f;
      perspective_projection_.near_plane = near_plane;
      perspective_projection_.far_plane = far_plane;
      return true;
    }
    return false;
  }

  // glDrawElements
  bool Draw(Matrix3 const&, DrawMesh const&);

  // draw points
  bool Draw(Matrix3 const&, Vector3 const* points, int num_points);

  // return result in RGB8 format
  uint8_t const* EndScene();
};

#endif