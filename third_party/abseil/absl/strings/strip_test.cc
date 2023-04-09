// Copyright 2017 The Abseil Authors.
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

// This file contains functions that remove a defined part from the string,
// i.e., strip the string.

#include "absl/strings/strip.h"

#include <cassert>
#include <cstdio>
#include <cstring>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/string_view.h"

namespace {

TEST(Strip, ConsumePrefixOneChar) {
  abslx::string_view input("abc");
  EXPECT_TRUE(abslx::ConsumePrefix(&input, "a"));
  EXPECT_EQ(input, "bc");

  EXPECT_FALSE(abslx::ConsumePrefix(&input, "x"));
  EXPECT_EQ(input, "bc");

  EXPECT_TRUE(abslx::ConsumePrefix(&input, "b"));
  EXPECT_EQ(input, "c");

  EXPECT_TRUE(abslx::ConsumePrefix(&input, "c"));
  EXPECT_EQ(input, "");

  EXPECT_FALSE(abslx::ConsumePrefix(&input, "a"));
  EXPECT_EQ(input, "");
}

TEST(Strip, ConsumePrefix) {
  abslx::string_view input("abcdef");
  EXPECT_FALSE(abslx::ConsumePrefix(&input, "abcdefg"));
  EXPECT_EQ(input, "abcdef");

  EXPECT_FALSE(abslx::ConsumePrefix(&input, "abce"));
  EXPECT_EQ(input, "abcdef");

  EXPECT_TRUE(abslx::ConsumePrefix(&input, ""));
  EXPECT_EQ(input, "abcdef");

  EXPECT_FALSE(abslx::ConsumePrefix(&input, "abcdeg"));
  EXPECT_EQ(input, "abcdef");

  EXPECT_TRUE(abslx::ConsumePrefix(&input, "abcdef"));
  EXPECT_EQ(input, "");

  input = "abcdef";
  EXPECT_TRUE(abslx::ConsumePrefix(&input, "abcde"));
  EXPECT_EQ(input, "f");
}

TEST(Strip, ConsumeSuffix) {
  abslx::string_view input("abcdef");
  EXPECT_FALSE(abslx::ConsumeSuffix(&input, "abcdefg"));
  EXPECT_EQ(input, "abcdef");

  EXPECT_TRUE(abslx::ConsumeSuffix(&input, ""));
  EXPECT_EQ(input, "abcdef");

  EXPECT_TRUE(abslx::ConsumeSuffix(&input, "def"));
  EXPECT_EQ(input, "abc");

  input = "abcdef";
  EXPECT_FALSE(abslx::ConsumeSuffix(&input, "abcdeg"));
  EXPECT_EQ(input, "abcdef");

  EXPECT_TRUE(abslx::ConsumeSuffix(&input, "f"));
  EXPECT_EQ(input, "abcde");

  EXPECT_TRUE(abslx::ConsumeSuffix(&input, "abcde"));
  EXPECT_EQ(input, "");
}

TEST(Strip, StripPrefix) {
  const abslx::string_view null_str;

  EXPECT_EQ(abslx::StripPrefix("foobar", "foo"), "bar");
  EXPECT_EQ(abslx::StripPrefix("foobar", ""), "foobar");
  EXPECT_EQ(abslx::StripPrefix("foobar", null_str), "foobar");
  EXPECT_EQ(abslx::StripPrefix("foobar", "foobar"), "");
  EXPECT_EQ(abslx::StripPrefix("foobar", "bar"), "foobar");
  EXPECT_EQ(abslx::StripPrefix("foobar", "foobarr"), "foobar");
  EXPECT_EQ(abslx::StripPrefix("", ""), "");
}

TEST(Strip, StripSuffix) {
  const abslx::string_view null_str;

  EXPECT_EQ(abslx::StripSuffix("foobar", "bar"), "foo");
  EXPECT_EQ(abslx::StripSuffix("foobar", ""), "foobar");
  EXPECT_EQ(abslx::StripSuffix("foobar", null_str), "foobar");
  EXPECT_EQ(abslx::StripSuffix("foobar", "foobar"), "");
  EXPECT_EQ(abslx::StripSuffix("foobar", "foo"), "foobar");
  EXPECT_EQ(abslx::StripSuffix("foobar", "ffoobar"), "foobar");
  EXPECT_EQ(abslx::StripSuffix("", ""), "");
}

TEST(Strip, RemoveExtraAsciiWhitespace) {
  const char* inputs[] = {
      "No extra space",
      "  Leading whitespace",
      "Trailing whitespace  ",
      "  Leading and trailing  ",
      " Whitespace \t  in\v   middle  ",
      "'Eeeeep!  \n Newlines!\n",
      "nospaces",
  };
  const char* outputs[] = {
      "No extra space",
      "Leading whitespace",
      "Trailing whitespace",
      "Leading and trailing",
      "Whitespace in middle",
      "'Eeeeep! Newlines!",
      "nospaces",
  };
  int NUM_TESTS = 7;

  for (int i = 0; i < NUM_TESTS; i++) {
    std::string s(inputs[i]);
    abslx::RemoveExtraAsciiWhitespace(&s);
    EXPECT_STREQ(outputs[i], s.c_str());
  }

  // Test that abslx::RemoveExtraAsciiWhitespace returns immediately for empty
  // strings (It was adding the \0 character to the C++ std::string, which broke
  // tests involving empty())
  std::string zero_string = "";
  assert(zero_string.empty());
  abslx::RemoveExtraAsciiWhitespace(&zero_string);
  EXPECT_EQ(zero_string.size(), 0);
  EXPECT_TRUE(zero_string.empty());
}

TEST(Strip, StripTrailingAsciiWhitespace) {
  std::string test = "foo  ";
  abslx::StripTrailingAsciiWhitespace(&test);
  EXPECT_EQ(test, "foo");

  test = "   ";
  abslx::StripTrailingAsciiWhitespace(&test);
  EXPECT_EQ(test, "");

  test = "";
  abslx::StripTrailingAsciiWhitespace(&test);
  EXPECT_EQ(test, "");

  test = " abc\t";
  abslx::StripTrailingAsciiWhitespace(&test);
  EXPECT_EQ(test, " abc");
}

TEST(String, StripLeadingAsciiWhitespace) {
  abslx::string_view orig = "\t  \n\f\r\n\vfoo";
  EXPECT_EQ("foo", abslx::StripLeadingAsciiWhitespace(orig));
  orig = "\t  \n\f\r\v\n\t  \n\f\r\v\n";
  EXPECT_EQ(abslx::string_view(), abslx::StripLeadingAsciiWhitespace(orig));
}

TEST(Strip, StripAsciiWhitespace) {
  std::string test2 = "\t  \f\r\n\vfoo \t\f\r\v\n";
  abslx::StripAsciiWhitespace(&test2);
  EXPECT_EQ(test2, "foo");
  std::string test3 = "bar";
  abslx::StripAsciiWhitespace(&test3);
  EXPECT_EQ(test3, "bar");
  std::string test4 = "\t  \f\r\n\vfoo";
  abslx::StripAsciiWhitespace(&test4);
  EXPECT_EQ(test4, "foo");
  std::string test5 = "foo \t\f\r\v\n";
  abslx::StripAsciiWhitespace(&test5);
  EXPECT_EQ(test5, "foo");
  abslx::string_view test6("\t  \f\r\n\vfoo \t\f\r\v\n");
  test6 = abslx::StripAsciiWhitespace(test6);
  EXPECT_EQ(test6, "foo");
  test6 = abslx::StripAsciiWhitespace(test6);
  EXPECT_EQ(test6, "foo");  // already stripped
}

}  // namespace
