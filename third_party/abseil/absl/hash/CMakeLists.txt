#
# Copyright 2018 The Abseil Authors.
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
    hash
  HDRS
    "hash.h"
  SRCS
    "internal/hash.cc"
    "internal/hash.h"
  COPTS
    ${ABSL_DEFAULT_COPTS}
    DEPS
    abslx::city
    abslx::config
    abslx::core_headers
    abslx::endian
    abslx::fixed_array
    abslx::meta
    abslx::int128
    abslx::strings
    abslx::optional
    abslx::variant
    abslx::utility
    abslx::wyhash
  PUBLIC
)

absl_cc_library(
  NAME
    hash_testing
  HDRS
    "hash_testing.h"
  COPTS
    ${ABSL_TEST_COPTS}
  DEPS
    abslx::spy_hash_state
    abslx::meta
    abslx::strings
    abslx::variant
    gmock
  TESTONLY
)

absl_cc_test(
  NAME
    hash_test
  SRCS
    "hash_test.cc"
  COPTS
    ${ABSL_TEST_COPTS}
  DEPS
    abslx::cord_test_helpers
    abslx::hash
    abslx::hash_testing
    abslx::core_headers
    abslx::flat_hash_set
    abslx::spy_hash_state
    abslx::meta
    abslx::int128
    gmock_main
)

absl_cc_library(
  NAME
    spy_hash_state
  HDRS
    "internal/spy_hash_state.h"
  COPTS
    ${ABSL_DEFAULT_COPTS}
  DEPS
    abslx::hash
    abslx::strings
    abslx::str_format
  TESTONLY
)

absl_cc_library(
  NAME
    city
  HDRS
    "internal/city.h"
  SRCS
    "internal/city.cc"
  COPTS
    ${ABSL_DEFAULT_COPTS}
  DEPS
    abslx::config
    abslx::core_headers
    abslx::endian
)

absl_cc_test(
  NAME
    city_test
  SRCS
    "internal/city_test.cc"
  COPTS
    ${ABSL_TEST_COPTS}
  DEPS
    abslx::city
    gmock_main
)

absl_cc_library(
  NAME
    wyhash
  HDRS
    "internal/wyhash.h"
  SRCS
    "internal/wyhash.cc"
  COPTS
    ${ABSL_DEFAULT_COPTS}
  DEPS
    abslx::config
    abslx::endian
    abslx::int128
)

absl_cc_test(
  NAME
    wyhash_test
  SRCS
    "internal/wyhash_test.cc"
  COPTS
    ${ABSL_TEST_COPTS}
  DEPS
    abslx::wyhash
    abslx::strings
    gmock_main
)
