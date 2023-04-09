#
# Copyright 2020 The Abseil Authors.
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
    status
  HDRS
    "status.h"
  SRCS
    "internal/status_internal.h"
    "status.cc"
    "status_payload_printer.h"
    "status_payload_printer.cc"
  COPTS
    ${ABSL_DEFAULT_COPTS}
  DEPS
    abslx::atomic_hook
    abslx::config
    abslx::core_headers
    abslx::raw_logging_internal
    abslx::inlined_vector
    abslx::stacktrace
    abslx::symbolize
    abslx::strings
    abslx::cord
    abslx::str_format
    abslx::optional
  PUBLIC
)

absl_cc_test(
  NAME
    status_test
  SRCS
   "status_test.cc"
  COPTS
    ${ABSL_TEST_COPTS}
  DEPS
    abslx::status
    abslx::strings
    gmock_main
)

absl_cc_library(
  NAME
    statusor
  HDRS
    "statusor.h"
  SRCS
    "statusor.cc"
    "internal/statusor_internal.h"
  COPTS
    ${ABSL_DEFAULT_COPTS}
  DEPS
    abslx::status
    abslx::core_headers
    abslx::raw_logging_internal
    abslx::type_traits
    abslx::strings
    abslx::utility
    abslx::variant
  PUBLIC
)

absl_cc_test(
  NAME
    statusor_test
  SRCS
   "statusor_test.cc"
  COPTS
    ${ABSL_TEST_COPTS}
  DEPS
    abslx::status
    abslx::statusor
    gmock_main
)
