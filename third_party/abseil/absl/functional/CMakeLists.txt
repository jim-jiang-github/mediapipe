#
# Copyright 2019 The Abseil Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

absl_cc_library(
  NAME
    bind_front
  SRCS
    "internal/front_binder.h"
  HDRS
    "bind_front.h"
  COPTS
    ${ABSL_DEFAULT_COPTS}
  DEPS
    abslx::base_internal
    abslx::compressed_tuple
  PUBLIC
)

absl_cc_test(
  NAME
    bind_front_test
  SRCS
    "bind_front_test.cc"
  COPTS
    ${ABSL_DEFAULT_COPTS}
  DEPS
    abslx::bind_front
    abslx::memory
    gmock_main
)

absl_cc_library(
  NAME
    function_ref
  SRCS
    "internal/function_ref.h"
  HDRS
    "function_ref.h"
  COPTS
    ${ABSL_DEFAULT_COPTS}
  DEPS
    abslx::base_internal
    abslx::meta
  PUBLIC
)

absl_cc_test(
  NAME
    function_ref_test
  SRCS
    "function_ref_test.cc"
  COPTS
    ${ABSL_TEST_COPTS}
  DEPS
    abslx::function_ref
    abslx::memory
    abslx::test_instance_tracker
    gmock_main
)
