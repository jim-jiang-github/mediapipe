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

#include "absl/strings/match.h"

#include "gtest/gtest.h"

namespace {

TEST(MatchTest, StartsWith) {
  const std::string s1("123\0abc", 7);
  const abslx::string_view a("foobar");
  const abslx::string_view b(s1);
  const abslx::string_view e;
  EXPECT_TRUE(abslx::StartsWith(a, a));
  EXPECT_TRUE(abslx::StartsWith(a, "foo"));
  EXPECT_TRUE(abslx::StartsWith(a, e));
  EXPECT_TRUE(abslx::StartsWith(b, s1));
  EXPECT_TRUE(abslx::StartsWith(b, b));
  EXPECT_TRUE(abslx::StartsWith(b, e));
  EXPECT_TRUE(abslx::StartsWith(e, ""));
  EXPECT_FALSE(abslx::StartsWith(a, b));
  EXPECT_FALSE(abslx::StartsWith(b, a));
  EXPECT_FALSE(abslx::StartsWith(e, a));
}

TEST(MatchTest, EndsWith) {
  const std::string s1("123\0abc", 7);
  const abslx::string_view a("foobar");
  const abslx::string_view b(s1);
  const abslx::string_view e;
  EXPECT_TRUE(abslx::EndsWith(a, a));
  EXPECT_TRUE(abslx::EndsWith(a, "bar"));
  EXPECT_TRUE(abslx::EndsWith(a, e));
  EXPECT_TRUE(abslx::EndsWith(b, s1));
  EXPECT_TRUE(abslx::EndsWith(b, b));
  EXPECT_TRUE(abslx::EndsWith(b, e));
  EXPECT_TRUE(abslx::EndsWith(e, ""));
  EXPECT_FALSE(abslx::EndsWith(a, b));
  EXPECT_FALSE(abslx::EndsWith(b, a));
  EXPECT_FALSE(abslx::EndsWith(e, a));
}

TEST(MatchTest, Contains) {
  abslx::string_view a("abcdefg");
  abslx::string_view b("abcd");
  abslx::string_view c("efg");
  abslx::string_view d("gh");
  EXPECT_TRUE(abslx::StrContains(a, a));
  EXPECT_TRUE(abslx::StrContains(a, b));
  EXPECT_TRUE(abslx::StrContains(a, c));
  EXPECT_FALSE(abslx::StrContains(a, d));
  EXPECT_TRUE(abslx::StrContains("", ""));
  EXPECT_TRUE(abslx::StrContains("abc", ""));
  EXPECT_FALSE(abslx::StrContains("", "a"));
}

TEST(MatchTest, ContainsChar) {
  abslx::string_view a("abcdefg");
  abslx::string_view b("abcd");
  EXPECT_TRUE(abslx::StrContains(a, 'a'));
  EXPECT_TRUE(abslx::StrContains(a, 'b'));
  EXPECT_TRUE(abslx::StrContains(a, 'e'));
  EXPECT_FALSE(abslx::StrContains(a, 'h'));

  EXPECT_TRUE(abslx::StrContains(b, 'a'));
  EXPECT_TRUE(abslx::StrContains(b, 'b'));
  EXPECT_FALSE(abslx::StrContains(b, 'e'));
  EXPECT_FALSE(abslx::StrContains(b, 'h'));

  EXPECT_FALSE(abslx::StrContains("", 'a'));
  EXPECT_FALSE(abslx::StrContains("", 'a'));
}

TEST(MatchTest, ContainsNull) {
  const std::string s = "foo";
  const char* cs = "foo";
  const abslx::string_view sv("foo");
  const abslx::string_view sv2("foo\0bar", 4);
  EXPECT_EQ(s, "foo");
  EXPECT_EQ(sv, "foo");
  EXPECT_NE(sv2, "foo");
  EXPECT_TRUE(abslx::EndsWith(s, sv));
  EXPECT_TRUE(abslx::StartsWith(cs, sv));
  EXPECT_TRUE(abslx::StrContains(cs, sv));
  EXPECT_FALSE(abslx::StrContains(cs, sv2));
}

TEST(MatchTest, EqualsIgnoreCase) {
  std::string text = "the";
  abslx::string_view data(text);

  EXPECT_TRUE(abslx::EqualsIgnoreCase(data, "The"));
  EXPECT_TRUE(abslx::EqualsIgnoreCase(data, "THE"));
  EXPECT_TRUE(abslx::EqualsIgnoreCase(data, "the"));
  EXPECT_FALSE(abslx::EqualsIgnoreCase(data, "Quick"));
  EXPECT_FALSE(abslx::EqualsIgnoreCase(data, "then"));
}

TEST(MatchTest, StartsWithIgnoreCase) {
  EXPECT_TRUE(abslx::StartsWithIgnoreCase("foo", "foo"));
  EXPECT_TRUE(abslx::StartsWithIgnoreCase("foo", "Fo"));
  EXPECT_TRUE(abslx::StartsWithIgnoreCase("foo", ""));
  EXPECT_FALSE(abslx::StartsWithIgnoreCase("foo", "fooo"));
  EXPECT_FALSE(abslx::StartsWithIgnoreCase("", "fo"));
}

TEST(MatchTest, EndsWithIgnoreCase) {
  EXPECT_TRUE(abslx::EndsWithIgnoreCase("foo", "foo"));
  EXPECT_TRUE(abslx::EndsWithIgnoreCase("foo", "Oo"));
  EXPECT_TRUE(abslx::EndsWithIgnoreCase("foo", ""));
  EXPECT_FALSE(abslx::EndsWithIgnoreCase("foo", "fooo"));
  EXPECT_FALSE(abslx::EndsWithIgnoreCase("", "fo"));
}

}  // namespace
