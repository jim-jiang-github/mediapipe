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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_FILE_SYSTEM_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_FILE_SYSTEM_UTILS_H_

#include <initializer_list>
#include <string>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorflow/core/platform/platform.h"

#ifdef PLATFORM_WINDOWS
const abslx::string_view kPathSep = "\\";
#else
const abslx::string_view kPathSep = "/";
#endif

namespace tensorflow {
namespace profiler {

inline std::string ProfilerJoinPathImpl(
    std::initializer_list<abslx::string_view> paths) {
  std::string result;
  for (abslx::string_view path : paths) {
    if (path.empty()) continue;

    if (result.empty()) {
      result = std::string(path);
      continue;
    }

    path = abslx::StripPrefix(path, kPathSep);
    if (abslx::EndsWith(result, kPathSep)) {
      abslx::StrAppend(&result, path);
    } else {
      abslx::StrAppend(&result, kPathSep, path);
    }
  }

  return result;
}

// A local duplication of ::tensorflow::io::JoinPath that supports windows.
// TODO(b/150699701): revert to use ::tensorflow::io::JoinPath when fixed.
template <typename... T>
std::string ProfilerJoinPath(const T&... args) {
  return ProfilerJoinPathImpl({args...});
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_FILE_SYSTEM_UTILS_H_
