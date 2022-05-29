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

#include "opengl_renderer.h"

#ifdef _WIN32
#include <windows.h> // must have this before include <gl.h>

// move to project settings (linker)?
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "User32.lib") // GetDC(), ReleaseDC()
#pragma comment(lib, "gdi32.lib") // ChoosePixelFormat()
#endif

#include "GL/glew.h" // must include <glew.h> before <gl.h>
#include <gl/GL.h>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace {

constexpr int GL_VERTEX_ATTRIBUTE_POSITION = 0;
constexpr int GL_VERTEX_ATTRIBUTE_TEXCOORD = 1;

constexpr char const* textured_fullscreen_vertex_program = R"(
  #version 400 core
  layout(location=0) in vec2 position;
  out vec2 tc_;
  void main() {
    gl_Position = vec4(200.0*position-100.0, 100.0, 100.0);
    tc_ = position;
  }
)";

constexpr char const* textured_model_vertex_program = R"(
  #version 400 core
  uniform mat4 matWorldViewProj;
  layout(location=0) in vec4 position;
  layout(location=1) in vec2 texcoord;
  out vec2 tc_;
  void main() {
    gl_Position = position * matWorldViewProj;
    tc_ = texcoord;
  }
)";

char const* simple_textured_fragment_program = R"(
  #version 400 core
  uniform sampler2D texture;
  in mediump vec2 tc_;
  out vec4 c0;
  void main() {
    c0 = texture2D(texture, tc_);
  }
)";

constexpr char const* solid_color_model_vertex_program = R"(
  #version 400 core
  uniform mat4 matWorldViewProj;
  layout(location=0) in vec4 position;
  void main() {
    gl_Position = position * matWorldViewProj;
  }
)";

char const* solid_color_fragment_program = R"(
  #version 400 core
  uniform vec4 color;
  out vec4 c0;
  void main() {
    c0 = color;
  }
)";

GLuint create_shader(GLenum type, char const* source, GLint const* length=nullptr) {
  GLuint shader = glCreateShader(type);
  if (shader) {
    glShaderSource(shader, 1, &source, length);
    glCompileShader(shader);

    // check errors...
    GLint status = GL_NO_ERROR;

#ifdef _DEBUG
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &status);
    if (status>1) {
      char* infoLog = new char[status+1];
      glGetShaderInfoLog(shader, status, NULL, infoLog);
      printf("shader compile log :\n%s\n", infoLog);
      delete[] infoLog;
    }
#endif

    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status) {
      return shader;
    }

    glDeleteShader(shader);
  }

  return 0;
}

GLuint create_program(char const* vs, char const* ps) {
  GLuint vertex_shader = create_shader(GL_VERTEX_SHADER, vs);
  if (vertex_shader) {
    GLuint fragment_shader = create_shader(GL_FRAGMENT_SHADER, ps); 
    if (fragment_shader) {
      GLuint program = glCreateProgram();
      if (program) {
        glAttachShader(program, vertex_shader);
        glAttachShader(program, fragment_shader);

        // Explicitly bind vertex attribute. Do this before glLinkProgram().
        glBindAttribLocation(program, GL_VERTEX_ATTRIBUTE_POSITION, "position");
        glBindAttribLocation(program, GL_VERTEX_ATTRIBUTE_TEXCOORD, "texcoord");

        // Link the program
        // Active attributes that are not explicitly bound will be bound by the linker when glLinkProgram is called.
        // hope we had explicit bound all vertex attributes
        glLinkProgram(program);

        // shaders not needed
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);

        // Check the link status
        GLint linked = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &linked);
        if (linked) {
          return program;
        }

        GLint infoLen = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen>1) {
          char* infoLog = new char[infoLen+1];
          glGetProgramInfoLog(program, infoLen, NULL, infoLog);
          printf("Error linking program:\n%s\n", infoLog);
          delete[] infoLog;
        }

        glDeleteProgram(program);
        return 0;
      }

      glDeleteShader(fragment_shader);
    }
    glDeleteShader(vertex_shader);
  }

  return 0;
}

} // namsepace {}

bool DrawMesh::Create(Mesh const& mesh, char const* texture_file) {
  Release();

  num_vertices_ = (int) mesh.vertices.size();
  num_triangles_ = (int) mesh.indices.size();
  if (num_vertices_<3 || 0==num_triangles_ || 0!=(num_triangles_%3)) {
    num_vertices_ = num_triangles_ = 0;
    return false;
  }

  transparent_ = false;
  if (texture_file) {
    cv::Mat tex = cv::imread(texture_file);
    if (!tex.empty() && (3==tex.channels() || 4==tex.channels() || 1==tex.channels())) {
      glGenTextures(1, &texture_);
      glBindTexture(GL_TEXTURE_2D, texture_);

      if (3==tex.channels()) {
        cv::cvtColor(tex, tex, cv::COLOR_BGR2RGB);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, tex.cols, tex.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, tex.data);
      } else if (4==tex.channels()) {
        transparent_ = true;
        cv::cvtColor(tex, tex, cv::COLOR_BGRA2RGBA);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, tex.cols, tex.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex.data);
      } else {
        cv::cvtColor(tex, tex, cv::COLOR_GRAY2RGB);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, tex.cols, tex.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, tex.data);
      }

      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
      glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
      glBindTexture(GL_TEXTURE_2D, 0);
    } else {
      num_vertices_ = num_triangles_ = 0;
      return false;
    }
  }

  std::vector<int> line_list;
  num_lines_ = GerenateLineListFromTriangleList(line_list, mesh.indices);
  num_triangles_ /= 3;

  glGenBuffers(2, bos_);

  // non-interleaved vertex buffer, so we can change vertex position
  glBindBuffer(GL_ARRAY_BUFFER, bos_[0]);
  {
    // interleaved format is better anyway.
    // see DrawMesh::Update() for why we do this.
    std::vector<float> de_interleave(num_vertices_*5);
    float* const vtx_data = de_interleave.data();
    float* xyz = vtx_data;
    float* uv = vtx_data + num_vertices_*3;
    for (auto const& v : mesh.vertices) {
      *xyz++ = v.x;
      *xyz++ = v.y;
      *xyz++ = v.z;
      *uv++ = v.u;
      *uv++ = v.v;
    }
    glBufferData(GL_ARRAY_BUFFER, num_vertices_*20, vtx_data, GL_STATIC_DRAW);
  }

  // indices
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bos_[1]);
  int const total_indices = num_triangles_*3 + num_lines_*2;
  if (num_vertices_>255) {
    std::vector<uint16_t> ib;
    ib.reserve(total_indices);
    for (int id : mesh.indices) {
      assert(id<num_vertices_);
      ib.push_back((uint16_t)id);
    }
    for (int id : line_list) {
      assert(id<num_vertices_);
      ib.push_back((uint16_t)id);
    }
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, total_indices*2, ib.data(), GL_STATIC_DRAW);
  } else {
    std::vector<uint8_t> ib;
    ib.reserve(total_indices);
    for (int id : mesh.indices) {
      assert(id<num_vertices_);
      ib.push_back((uint8_t)id);
    }
    for (int id : line_list) {
      assert(id<num_vertices_);
      ib.push_back((uint8_t)id);
    }
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, total_indices, ib.data(), GL_STATIC_DRAW);
  }

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  color_disable_ = false;
  depth_disable_ = false;
  depth_offset_ = 0;

  return true;
}

bool DrawMesh::Update(Vector3 const* vertices, int size) {
  if (vertices && size<=num_vertices_ && bos_[0]) {
    glBindBuffer(GL_ARRAY_BUFFER, bos_[0]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size*12, vertices);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return true;
  }
  return false;
}

void DrawMesh::Release() {
  if (texture_) {
    glDeleteTextures(1, &texture_);
    texture_ = 0;
  }

  if (bos_[0]) {
    glDeleteBuffers(2, bos_);
    bos_[0] = bos_[1] = 0;
  }

  num_vertices_ = num_triangles_ = num_lines_ = 0;

  color_disable_ = false;
  depth_disable_ = false;
  depth_offset_ = 0;
}

OpenGLRenderer::OpenGLRenderer():
read_pixels_(nullptr),
framebuffer_(0),
fullscren_shader_(0),
fullscreen_texture_(0),
textured_model_shader_(0),
color_model_shader_(0),
textured_model_shader_mvp_loc_(-1),
color_model_shader_mvp_loc_(-1),
color_model_shader_color_loc_(-1),
width_(0),height_(0) {
  renderbuffers_[0] = renderbuffers_[1] = 0;
}

bool OpenGLRenderer::InitGLObjects_() {
#ifdef __GLEW_H__
  if (GLEW_OK!=glewInit()) {
    return false;
  }
#endif

  glShadeModel(GL_SMOOTH);
  glClearColor(1.0f, 0.0f, 1.0f, 0.0f);
  glClearDepth(1.0f);
  glDepthRange(0.0f, 1.0f);
  glDepthMask(GL_FALSE);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glFrontFace(GL_CCW);

  // increase rasterized diameter of points
  glDisable(GL_PROGRAM_POINT_SIZE);
  glPointSize(4.0f);

  fullscren_shader_ = create_program(textured_fullscreen_vertex_program, simple_textured_fragment_program);
  auto texture_loc = glGetUniformLocation(fullscren_shader_, "texture");
  glProgramUniform1i(fullscren_shader_, texture_loc, 0);

  textured_model_shader_ = create_program(textured_model_vertex_program, simple_textured_fragment_program);
  texture_loc = glGetUniformLocation(textured_model_shader_, "texture");
  glProgramUniform1i(textured_model_shader_, texture_loc, 0);
  textured_model_shader_mvp_loc_ = glGetUniformLocation(textured_model_shader_, "matWorldViewProj");

  color_model_shader_ = create_program(solid_color_model_vertex_program, solid_color_fragment_program);
  color_model_shader_mvp_loc_ = glGetUniformLocation(color_model_shader_, "matWorldViewProj");
  color_model_shader_color_loc_= glGetUniformLocation(color_model_shader_, "color");

  glGenFramebuffers(1, &framebuffer_);
  glGenRenderbuffers(2, renderbuffers_);

  glGenTextures(1, &fullscreen_texture_);
  glBindTexture(GL_TEXTURE_2D, fullscreen_texture_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glBindTexture(GL_TEXTURE_2D, 0);

  width_ = height_ = 0;
  return GL_NO_ERROR==glGetError();
}

void OpenGLRenderer::DestroyGLObjects_() {
  if (fullscren_shader_) {
    glDeleteProgram(fullscren_shader_);
    fullscren_shader_ = 0;
  }

  if (fullscreen_texture_) {
    glDeleteTextures(1, &fullscreen_texture_);
    fullscreen_texture_ = 0;
  }

  glDeleteRenderbuffers(2, renderbuffers_);
  renderbuffers_[0] = renderbuffers_[1] = 0;
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  if (framebuffer_) {
    glDeleteRenderbuffers(1, &framebuffer_);
    framebuffer_ = 0;
  }

  if (read_pixels_) {
    free(read_pixels_);
    read_pixels_ = nullptr;
  }
  width_ = height_ = 0;
}

bool OpenGLRenderer::Run_() {
  run_status_ = -1;

#ifdef _WIN32
  render_thread_ = std::move(std::thread([this] {
    HWND hWnd = ::CreateWindowA("STATIC", "dummy-invisible", WS_DISABLED, 0, 0, 100, 100, NULL, NULL, NULL, NULL);
    if (NULL==hWnd) {
      printf("ERROR: [OpenGL] failed to create dummy window\n");
      run_status_ = -100;
      return;
    }
    auto hDC = GetDC(hWnd);
    if (NULL==hDC) {
      printf("ERROR: [OpenGL] failed to get DC from dummy window\n");
      DestroyWindow(hWnd);
      run_status_ = -101;
      return;
    }

    PIXELFORMATDESCRIPTOR pfd = {
      sizeof(PIXELFORMATDESCRIPTOR),
      1,
      PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
      PFD_TYPE_RGBA,
      24, // color depth
      0, 0, 0, 0, 0, 0,
      0,
      0,
      0,
      0, 0, 0, 0,
      16, // 16Bit Zbuffer
      0,
      0,
      PFD_MAIN_PLANE,
      0,
      0, 0, 0
    };
    PIXELFORMATDESCRIPTOR pfd2; 
    auto pixel_format = ChoosePixelFormat(hDC, &pfd);
    if (pixel_format && DescribePixelFormat(hDC, pixel_format, sizeof(pfd2), &pfd2) &&
        SetPixelFormat(hDC, pixel_format, &pfd2)) {
    } else {
      printf("ERROR: [OpenGL] No compatible pixel formats for Window DC\n");
      ReleaseDC(hWnd, hDC);
      DestroyWindow(hWnd);
      run_status_ = -102;
      return;
    }

    auto hRC = wglCreateContext(hDC);
    if (hRC) {
      wglMakeCurrent(hDC, hRC);

      if (InitGLObjects_()) {
        printf("INFO: [OpenGL] ready\n");
        if (OnInitedGL_()) {
          run_status_ = 0;
          while (run_status_>=0) {
            std::unique_lock<std::mutex> lock(render_wait_mutex_);
            render_wait_cond_var_.wait(lock);

            DrawFrame_();

            if (run_status_>0) {
              run_status_ = 0;
            }
            render_done_cond_var_.notify_one();
          }

          OnDestroyGL_();
        }

        DestroyGLObjects_();
        printf("INFO: [OpenGL] stopped\n");
      } else {
        run_status_ = -104;
      }

      wglMakeCurrent(hDC, NULL);
      wglDeleteContext(hRC);
    } else {
      printf("ERROR: [OpenGL] Failed to create OpenGL rendering context Error:0x%X\n", GetLastError());
      run_status_ = -103;
    }

    ReleaseDC(hWnd, hDC);
    DestroyWindow(hWnd);
  }));
#else
#error "[OpenGL Renderer] other platforms to be implemented, EGL?"
  render_thread_ = std::move(std::thread([this] {
    run_status_ = -100;
  }));
#endif

  // TO-DO: use condition variable to wait...
  while (-1==run_status_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  if (run_status_>=0) {
    return true;
  }

  return false;
}

bool OpenGLRenderer::BeginScene(uint8_t const* background, int width, int height) {
  assert(background && width>0 && height>0);
  if ((width_*height_)<(width*height) || nullptr==read_pixels_) {
    read_pixels_ = (uint8_t*) malloc(width*height*4);
    if (!read_pixels_) {
      return false;
    }
  }

  if (width_!=width || height_!=height) {
    assert(GL_NO_ERROR==glGetError());
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
    glBindRenderbuffer(GL_RENDERBUFFER, renderbuffers_[0]);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height);
    // normally, we call glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, render_to_texture, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, renderbuffers_[0]);

    glBindRenderbuffer(GL_RENDERBUFFER, renderbuffers_[1]);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderbuffers_[1]);

    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (GL_FRAMEBUFFER_COMPLETE!=status) {
      printf("ERROR: [OpenGL] Framebuffer not complete! 0x%X\n", status);
      return false;
    }

    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer_);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer_);
    glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);

    glViewport(0, 0, width, height);

    //
    width_ = width;
    height_ = height;

    auto gl_error = glGetError();
    if (GL_NO_ERROR!=gl_error) {
      printf("ERROR: [OpenGL] failed to bind frame buffer 0x%X\n", gl_error);
      return false;
    }
  }

  // draw background - fullscreen quad with writing Z=1.0

  // no testing needed, every pixel pass!
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);

  // will write Z= 1.0
  glDepthMask(GL_TRUE);

#if 1
  // since we draw fullscreen with writing Z=1.0, then why should it need this clear?
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);
  glDepthMask(GL_FALSE); // if clear, need no z write
#endif

  glUseProgram(fullscren_shader_);

  // upload texture
  glActiveTexture(GL_TEXTURE0); // only use one texture unit 
  glBindTexture(GL_TEXTURE_2D, fullscreen_texture_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width_, height_, 0, GL_RGB, GL_UNSIGNED_BYTE, background);

  constexpr uint8_t fullscreen_vb[] = { 0,255,  0,0,  255,255,  255,0 }; // fullscreen quad
  glEnableVertexAttribArray(GL_VERTEX_ATTRIBUTE_POSITION);
  glVertexAttribPointer(GL_VERTEX_ATTRIBUTE_POSITION, 2, GL_UNSIGNED_BYTE, GL_TRUE, 0, fullscreen_vb);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  glUseProgram(0);

  //
  // build projection matrix wrt to vfov
  //
  // [Important NOTE]
  // To make processing pixels easier from glReadPixels (see below), here it negates m22 to
  // render it upside down. So it need to reverse the GL front face winding to be CW. For models,
  // the front face is still CCW.
  float const cot_half_fovY = 1.0f/tan(0.5f*perspective_projection_.vertical_fov);
  float const d_depth = 1.0f/(perspective_projection_.near_plane-perspective_projection_.far_plane);
  perspective_projection_.m11 = cot_half_fovY*(float)height/(float)width;
  perspective_projection_.m22 = -cot_half_fovY; // upside down!!!
  perspective_projection_.m33 = (perspective_projection_.near_plane+perspective_projection_.far_plane)*d_depth;
  perspective_projection_.m34 = 2.0f*perspective_projection_.near_plane*perspective_projection_.far_plane*d_depth;

  glEnable(GL_CULL_FACE);
  glFrontFace(GL_CW); // see above perspective_projection_.m22

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  //glDepthFunc(GL_LEQUAL);

  glDepthMask(GL_TRUE);

  return true;
}

bool OpenGLRenderer::Draw(Matrix3 const& xform, DrawMesh const& e) {
  // build matrix
  float mvp[16];
  perspective_projection_.MVP(mvp, xform);

  size_t buffer_offset;

  if (e.color_disable_) {
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

    if (e.depth_offset_) {
      glEnable(GL_POLYGON_OFFSET_FILL);
      glPolygonOffset(1.0f, (float) e.depth_offset_);
    }

    glUseProgram(color_model_shader_);
    glUniformMatrix4fv(color_model_shader_mvp_loc_, 1, GL_FALSE, mvp);

    glBindBuffer(GL_ARRAY_BUFFER, e.bos_[0]);
    glEnableVertexAttribArray(GL_VERTEX_ATTRIBUTE_POSITION);
    glVertexAttribPointer(GL_VERTEX_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, e.bos_[1]);
    glDrawElements(GL_TRIANGLES, e.num_triangles_*3, (e.num_vertices_>255) ? GL_UNSIGNED_SHORT:GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    if (e.depth_offset_) {
      glDisable(GL_POLYGON_OFFSET_FILL);
      glPolygonOffset(0.0f, 0.0f);
    }

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    return true;
  }

  if (e.texture_) {
    glUseProgram(textured_model_shader_);
    glUniformMatrix4fv(textured_model_shader_mvp_loc_, 1, GL_FALSE, mvp);
    glBindTexture(GL_TEXTURE_2D, e.texture_);
    if (e.transparent_) {
      // alpha blending
    }
  } else {
    glUseProgram(color_model_shader_);
    glUniformMatrix4fv(color_model_shader_mvp_loc_, 1, GL_FALSE, mvp);
    glUniform4fv(color_model_shader_color_loc_, 1, color_);

    if (color_[3]<1.0f) {
      // alpha blending
    }
  }

  // bind vertex and index buffers
  glBindBuffer(GL_ARRAY_BUFFER, e.bos_[0]);
  glEnableVertexAttribArray(GL_VERTEX_ATTRIBUTE_POSITION);
  glVertexAttribPointer(GL_VERTEX_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  if (e.texture_) {
    buffer_offset = e.num_vertices_*12;
    glEnableVertexAttribArray(GL_VERTEX_ATTRIBUTE_TEXCOORD);
    glVertexAttribPointer(GL_VERTEX_ATTRIBUTE_TEXCOORD, 2, GL_FLOAT, GL_FALSE, 0, (GLvoid const*) buffer_offset);
  } else {
    glDisableVertexAttribArray(GL_VERTEX_ATTRIBUTE_TEXCOORD);
  }

  // draw call
  if (0==e.mode_ || 3==e.mode_) {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, e.bos_[1]);
    glDrawElements(GL_TRIANGLES, e.num_triangles_*3, (e.num_vertices_>255) ? GL_UNSIGNED_SHORT:GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  } else if (2==e.mode_) {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, e.bos_[1]);
    buffer_offset = e.num_triangles_*((e.num_vertices_>255) ? 6:3); // with offset
    glDrawElements(GL_LINES, e.num_lines_*2, (e.num_vertices_>255) ? GL_UNSIGNED_SHORT:GL_UNSIGNED_BYTE, (GLvoid const*) buffer_offset);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  } else {
    glDrawArrays(GL_POINTS, 0, e.num_vertices_);
  }

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glUseProgram(0);

  return true;
}

bool OpenGLRenderer::Draw(Matrix3 const& xform, Vector3 const* points, int num_points) {
  float mvp[16];
  perspective_projection_.MVP(mvp, xform);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  glUseProgram(color_model_shader_);
  glUniformMatrix4fv(color_model_shader_mvp_loc_, 1, GL_FALSE, mvp);
  glUniform4fv(color_model_shader_color_loc_, 1, color_);

  glEnableVertexAttribArray(GL_VERTEX_ATTRIBUTE_POSITION);
  glVertexAttribPointer(GL_VERTEX_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, points);

  //glDisable(GL_DEPTH_TEST);
  glDrawArrays(GL_POINTS, 0, num_points);
  //glEnable(GL_DEPTH_TEST);

  glUseProgram(0);
  return true;
}

uint8_t const* OpenGLRenderer::EndScene() {
  if (read_pixels_) {
    // normally, I don't do read pixel and claim it's GPU accelerated.
    // but limited by mediapipe calculate graph, i have no other choices...
    glReadPixels(0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, read_pixels_);

    // render upside down, so just take out alpha channel
    int const total_pixels = width_*height_;
    uint8_t const* src = read_pixels_ + 4;
    uint8_t* dst = read_pixels_ + 3;
    for (int i=1; i<total_pixels; ++i) {
      *dst++ = *src++;
      *dst++ = *src++;
      *dst++ = *src++;
      ++src;
    }
  }

#ifdef _DEBUG
  // check OpenGL error
  auto gl_error = glGetError();
  if (GL_NO_ERROR!=gl_error) {
    printf("ERROR: [OpenGL] End Scene 0x%X\n", gl_error);
  }
#endif

  return read_pixels_;
}
