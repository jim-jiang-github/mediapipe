/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/gpu/common/task/texture2d_desc.h"

#include <string>

#include "absl/strings/str_cat.h"

namespace tflite {
namespace gpu {

void Texture2DDescriptor::Release() { data.clear(); }

GPUResources Texture2DDescriptor::GetGPUResources(
    const GpuInfo& gpu_info) const {
  GPUResources resources;
  GPUImage2DDescriptor desc;
  desc.data_type = element_type;
  desc.normalized = normalized;
  desc.normalized_type = normalized_type;
  desc.access_type = access_type_;
  resources.images2d.push_back({"tex2d", desc});
  if (gpu_info.IsApiOpenGl() && gpu_info.opengl_info.major_version < 3) {
    resources.floats.push_back("inv_tex_width");
    resources.floats.push_back("inv_tex_height");
  }
  return resources;
}

abslx::Status Texture2DDescriptor::PerformSelector(
    const GpuInfo& gpu_info, const std::string& selector,
    const std::vector<std::string>& args,
    const std::vector<std::string>& template_args, std::string* result) const {
  if (selector == "Read") {
    return PerformReadSelector(gpu_info, args, result);
  } else {
    return abslx::NotFoundError(abslx::StrCat(
        "Texture2DDescriptor don't have selector with name - ", selector));
  }
}

abslx::Status Texture2DDescriptor::PerformReadSelector(
    const GpuInfo& gpu_info, const std::vector<std::string>& args,
    std::string* result) const {
  if (args.size() != 2) {
    return abslx::NotFoundError(
        abslx::StrCat("Texture2DDescriptor Read require two arguments, but ",
                     args.size(), " was passed"));
  }
  if (gpu_info.IsApiMetal()) {
    *result =
        abslx::StrCat("tex2d.read(ushort2(", args[0], ", " + args[1] + "))");
    return abslx::OkStatus();
  } else if (gpu_info.IsApiOpenCl()) {
    std::string read;
    switch (element_type) {
      case DataType::FLOAT32:
        read = "read_imagef";
        break;
      case DataType::FLOAT16:
        read = "read_imageh";
        break;
      case DataType::INT8:
      case DataType::INT16:
      case DataType::INT32:
        if (normalized) {
          read = normalized_type == DataType::FLOAT16 ? "read_imageh"
                                                      : "read_imagef";
        } else {
          read = "read_imagei";
        }
        break;
      case DataType::UINT8:
      case DataType::UINT16:
      case DataType::UINT32:
        if (normalized) {
          read = normalized_type == DataType::FLOAT16 ? "read_imageh"
                                                      : "read_imagef";
        } else {
          read = "read_imageui";
        }
        break;
      default:
        read = "unknown_type";
        break;
    }
    *result = abslx::StrCat(read, "(tex2d, smp_none, (int2)(", args[0],
                           ", " + args[1] + "))");
    return abslx::OkStatus();
  } else if (gpu_info.IsGlsl()) {
    if (gpu_info.IsApiOpenGl() && gpu_info.opengl_info.major_version < 3) {
      *result = abslx::StrCat("texture2D(tex2d, vec2(float(", args[0],
                             ") * inv_tex_width, float(", args[1],
                             ") * inv_tex_height))");
      return abslx::OkStatus();
    } else {
      *result = "texelFetch(tex2d, ivec2(" + args[0] + ", " + args[1] + "), 0)";
      if (element_type == DataType::FLOAT16 &&
          gpu_info.IsGlslSupportsExplicitFp16()) {
        *result = "f16vec4(" + *result + ")";
      }
      return abslx::OkStatus();
    }
  } else {
    return abslx::UnimplementedError(
        "No implementation of Texture2D.Read for this API.");
  }
}

}  // namespace gpu
}  // namespace tflite
