/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/url.h"

#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace data {

URL::URL(abslx::string_view url) { Parse(url); }

void URL::Parse(abslx::string_view url) {
  // Parses `url` into host:port. The port can be a number, named port, or
  // dynamic port (i.e.: %port_name%).
  abslx::string_view regexp = "(.*):([a-zA-Z0-9_]+|%port(_[a-zA-Z0-9_]+)?%)";

  if (!RE2::FullMatch(url, regexp, &host_, &port_)) {
    host_ = std::string(url);
    port_ = "";
  }
}

}  // namespace data
}  // namespace tensorflow
