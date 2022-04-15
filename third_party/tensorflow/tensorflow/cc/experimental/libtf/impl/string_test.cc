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
#include "tensorflow/cc/experimental/libtf/impl/string.h"

#include "tensorflow/core/platform/test.h"

namespace tf {
namespace libtf {
namespace impl {

TEST(StringTest, TestBasicInterning) {
  String s1("foo");
  String s2("foo");
  EXPECT_EQ(&s1.str(), &s2.str());
}

TEST(StringTest, TestIOStream) {
  String s("foo");
  std::stringstream stream;
  stream << s;
  ASSERT_EQ(stream.str(), "foo");
}

}  // namespace impl
}  // namespace libtf
}  // namespace tf
