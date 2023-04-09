// Copyright 2018 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "absl/base/log_severity.h"

#include <cstdint>
#include <ios>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/internal/flag.h"
#include "absl/flags/marshalling.h"
#include "absl/strings/str_cat.h"

namespace {
using ::testing::Eq;
using ::testing::IsFalse;
using ::testing::IsTrue;
using ::testing::TestWithParam;
using ::testing::Values;

std::string StreamHelper(abslx::LogSeverity value) {
  std::ostringstream stream;
  stream << value;
  return stream.str();
}

TEST(StreamTest, Works) {
  EXPECT_THAT(StreamHelper(static_cast<abslx::LogSeverity>(-100)),
              Eq("abslx::LogSeverity(-100)"));
  EXPECT_THAT(StreamHelper(abslx::LogSeverity::kInfo), Eq("INFO"));
  EXPECT_THAT(StreamHelper(abslx::LogSeverity::kWarning), Eq("WARNING"));
  EXPECT_THAT(StreamHelper(abslx::LogSeverity::kError), Eq("ERROR"));
  EXPECT_THAT(StreamHelper(abslx::LogSeverity::kFatal), Eq("FATAL"));
  EXPECT_THAT(StreamHelper(static_cast<abslx::LogSeverity>(4)),
              Eq("abslx::LogSeverity(4)"));
}

static_assert(
    abslx::flags_internal::FlagUseOneWordStorage<abslx::LogSeverity>::value,
    "Flags of type abslx::LogSeverity ought to be lock-free.");

using ParseFlagFromOutOfRangeIntegerTest = TestWithParam<int64_t>;
INSTANTIATE_TEST_SUITE_P(
    Instantiation, ParseFlagFromOutOfRangeIntegerTest,
    Values(static_cast<int64_t>(std::numeric_limits<int>::min()) - 1,
           static_cast<int64_t>(std::numeric_limits<int>::max()) + 1));
TEST_P(ParseFlagFromOutOfRangeIntegerTest, ReturnsError) {
  const std::string to_parse = abslx::StrCat(GetParam());
  abslx::LogSeverity value;
  std::string error;
  EXPECT_THAT(abslx::ParseFlag(to_parse, &value, &error), IsFalse()) << value;
}

using ParseFlagFromAlmostOutOfRangeIntegerTest = TestWithParam<int>;
INSTANTIATE_TEST_SUITE_P(Instantiation,
                         ParseFlagFromAlmostOutOfRangeIntegerTest,
                         Values(std::numeric_limits<int>::min(),
                                std::numeric_limits<int>::max()));
TEST_P(ParseFlagFromAlmostOutOfRangeIntegerTest, YieldsExpectedValue) {
  const auto expected = static_cast<abslx::LogSeverity>(GetParam());
  const std::string to_parse = abslx::StrCat(GetParam());
  abslx::LogSeverity value;
  std::string error;
  ASSERT_THAT(abslx::ParseFlag(to_parse, &value, &error), IsTrue()) << error;
  EXPECT_THAT(value, Eq(expected));
}

using ParseFlagFromIntegerMatchingEnumeratorTest =
    TestWithParam<std::tuple<abslx::string_view, abslx::LogSeverity>>;
INSTANTIATE_TEST_SUITE_P(
    Instantiation, ParseFlagFromIntegerMatchingEnumeratorTest,
    Values(std::make_tuple("0", abslx::LogSeverity::kInfo),
           std::make_tuple(" 0", abslx::LogSeverity::kInfo),
           std::make_tuple("-0", abslx::LogSeverity::kInfo),
           std::make_tuple("+0", abslx::LogSeverity::kInfo),
           std::make_tuple("00", abslx::LogSeverity::kInfo),
           std::make_tuple("0 ", abslx::LogSeverity::kInfo),
           std::make_tuple("0x0", abslx::LogSeverity::kInfo),
           std::make_tuple("1", abslx::LogSeverity::kWarning),
           std::make_tuple("+1", abslx::LogSeverity::kWarning),
           std::make_tuple("2", abslx::LogSeverity::kError),
           std::make_tuple("3", abslx::LogSeverity::kFatal)));
TEST_P(ParseFlagFromIntegerMatchingEnumeratorTest, YieldsExpectedValue) {
  const abslx::string_view to_parse = std::get<0>(GetParam());
  const abslx::LogSeverity expected = std::get<1>(GetParam());
  abslx::LogSeverity value;
  std::string error;
  ASSERT_THAT(abslx::ParseFlag(to_parse, &value, &error), IsTrue()) << error;
  EXPECT_THAT(value, Eq(expected));
}

using ParseFlagFromOtherIntegerTest =
    TestWithParam<std::tuple<abslx::string_view, int>>;
INSTANTIATE_TEST_SUITE_P(Instantiation, ParseFlagFromOtherIntegerTest,
                         Values(std::make_tuple("-1", -1),
                                std::make_tuple("4", 4),
                                std::make_tuple("010", 10),
                                std::make_tuple("0x10", 16)));
TEST_P(ParseFlagFromOtherIntegerTest, YieldsExpectedValue) {
  const abslx::string_view to_parse = std::get<0>(GetParam());
  const auto expected = static_cast<abslx::LogSeverity>(std::get<1>(GetParam()));
  abslx::LogSeverity value;
  std::string error;
  ASSERT_THAT(abslx::ParseFlag(to_parse, &value, &error), IsTrue()) << error;
  EXPECT_THAT(value, Eq(expected));
}

using ParseFlagFromEnumeratorTest =
    TestWithParam<std::tuple<abslx::string_view, abslx::LogSeverity>>;
INSTANTIATE_TEST_SUITE_P(
    Instantiation, ParseFlagFromEnumeratorTest,
    Values(std::make_tuple("INFO", abslx::LogSeverity::kInfo),
           std::make_tuple("info", abslx::LogSeverity::kInfo),
           std::make_tuple("kInfo", abslx::LogSeverity::kInfo),
           std::make_tuple("iNfO", abslx::LogSeverity::kInfo),
           std::make_tuple("kInFo", abslx::LogSeverity::kInfo),
           std::make_tuple("WARNING", abslx::LogSeverity::kWarning),
           std::make_tuple("warning", abslx::LogSeverity::kWarning),
           std::make_tuple("kWarning", abslx::LogSeverity::kWarning),
           std::make_tuple("WaRnInG", abslx::LogSeverity::kWarning),
           std::make_tuple("KwArNiNg", abslx::LogSeverity::kWarning),
           std::make_tuple("ERROR", abslx::LogSeverity::kError),
           std::make_tuple("error", abslx::LogSeverity::kError),
           std::make_tuple("kError", abslx::LogSeverity::kError),
           std::make_tuple("eRrOr", abslx::LogSeverity::kError),
           std::make_tuple("kErRoR", abslx::LogSeverity::kError),
           std::make_tuple("FATAL", abslx::LogSeverity::kFatal),
           std::make_tuple("fatal", abslx::LogSeverity::kFatal),
           std::make_tuple("kFatal", abslx::LogSeverity::kFatal),
           std::make_tuple("FaTaL", abslx::LogSeverity::kFatal),
           std::make_tuple("KfAtAl", abslx::LogSeverity::kFatal)));
TEST_P(ParseFlagFromEnumeratorTest, YieldsExpectedValue) {
  const abslx::string_view to_parse = std::get<0>(GetParam());
  const abslx::LogSeverity expected = std::get<1>(GetParam());
  abslx::LogSeverity value;
  std::string error;
  ASSERT_THAT(abslx::ParseFlag(to_parse, &value, &error), IsTrue()) << error;
  EXPECT_THAT(value, Eq(expected));
}

using ParseFlagFromGarbageTest = TestWithParam<abslx::string_view>;
INSTANTIATE_TEST_SUITE_P(Instantiation, ParseFlagFromGarbageTest,
                         Values("", "\0", " ", "garbage", "kkinfo", "I"));
TEST_P(ParseFlagFromGarbageTest, ReturnsError) {
  const abslx::string_view to_parse = GetParam();
  abslx::LogSeverity value;
  std::string error;
  EXPECT_THAT(abslx::ParseFlag(to_parse, &value, &error), IsFalse()) << value;
}

using UnparseFlagToEnumeratorTest =
    TestWithParam<std::tuple<abslx::LogSeverity, abslx::string_view>>;
INSTANTIATE_TEST_SUITE_P(
    Instantiation, UnparseFlagToEnumeratorTest,
    Values(std::make_tuple(abslx::LogSeverity::kInfo, "INFO"),
           std::make_tuple(abslx::LogSeverity::kWarning, "WARNING"),
           std::make_tuple(abslx::LogSeverity::kError, "ERROR"),
           std::make_tuple(abslx::LogSeverity::kFatal, "FATAL")));
TEST_P(UnparseFlagToEnumeratorTest, ReturnsExpectedValueAndRoundTrips) {
  const abslx::LogSeverity to_unparse = std::get<0>(GetParam());
  const abslx::string_view expected = std::get<1>(GetParam());
  const std::string stringified_value = abslx::UnparseFlag(to_unparse);
  EXPECT_THAT(stringified_value, Eq(expected));
  abslx::LogSeverity reparsed_value;
  std::string error;
  EXPECT_THAT(abslx::ParseFlag(stringified_value, &reparsed_value, &error),
              IsTrue());
  EXPECT_THAT(reparsed_value, Eq(to_unparse));
}

using UnparseFlagToOtherIntegerTest = TestWithParam<int>;
INSTANTIATE_TEST_SUITE_P(Instantiation, UnparseFlagToOtherIntegerTest,
                         Values(std::numeric_limits<int>::min(), -1, 4,
                                std::numeric_limits<int>::max()));
TEST_P(UnparseFlagToOtherIntegerTest, ReturnsExpectedValueAndRoundTrips) {
  const abslx::LogSeverity to_unparse =
      static_cast<abslx::LogSeverity>(GetParam());
  const std::string expected = abslx::StrCat(GetParam());
  const std::string stringified_value = abslx::UnparseFlag(to_unparse);
  EXPECT_THAT(stringified_value, Eq(expected));
  abslx::LogSeverity reparsed_value;
  std::string error;
  EXPECT_THAT(abslx::ParseFlag(stringified_value, &reparsed_value, &error),
              IsTrue());
  EXPECT_THAT(reparsed_value, Eq(to_unparse));
}
}  // namespace
