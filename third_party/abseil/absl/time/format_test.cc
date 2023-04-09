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

#include <cstdint>
#include <limits>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/time/internal/test_util.h"
#include "absl/time/time.h"

using testing::HasSubstr;

namespace {

// A helper that tests the given format specifier by itself, and with leading
// and trailing characters.  For example: TestFormatSpecifier(t, "%a", "Thu").
void TestFormatSpecifier(abslx::Time t, abslx::TimeZone tz,
                         const std::string& fmt, const std::string& ans) {
  EXPECT_EQ(ans, abslx::FormatTime(fmt, t, tz));
  EXPECT_EQ("xxx " + ans, abslx::FormatTime("xxx " + fmt, t, tz));
  EXPECT_EQ(ans + " yyy", abslx::FormatTime(fmt + " yyy", t, tz));
  EXPECT_EQ("xxx " + ans + " yyy",
            abslx::FormatTime("xxx " + fmt + " yyy", t, tz));
}

//
// Testing FormatTime()
//

TEST(FormatTime, Basics) {
  abslx::TimeZone tz = abslx::UTCTimeZone();
  abslx::Time t = abslx::FromTimeT(0);

  // Starts with a couple basic edge cases.
  EXPECT_EQ("", abslx::FormatTime("", t, tz));
  EXPECT_EQ(" ", abslx::FormatTime(" ", t, tz));
  EXPECT_EQ("  ", abslx::FormatTime("  ", t, tz));
  EXPECT_EQ("xxx", abslx::FormatTime("xxx", t, tz));
  std::string big(128, 'x');
  EXPECT_EQ(big, abslx::FormatTime(big, t, tz));
  // Cause the 1024-byte buffer to grow.
  std::string bigger(100000, 'x');
  EXPECT_EQ(bigger, abslx::FormatTime(bigger, t, tz));

  t += abslx::Hours(13) + abslx::Minutes(4) + abslx::Seconds(5);
  t += abslx::Milliseconds(6) + abslx::Microseconds(7) + abslx::Nanoseconds(8);
  EXPECT_EQ("1970-01-01", abslx::FormatTime("%Y-%m-%d", t, tz));
  EXPECT_EQ("13:04:05", abslx::FormatTime("%H:%M:%S", t, tz));
  EXPECT_EQ("13:04:05.006", abslx::FormatTime("%H:%M:%E3S", t, tz));
  EXPECT_EQ("13:04:05.006007", abslx::FormatTime("%H:%M:%E6S", t, tz));
  EXPECT_EQ("13:04:05.006007008", abslx::FormatTime("%H:%M:%E9S", t, tz));
}

TEST(FormatTime, LocaleSpecific) {
  const abslx::TimeZone tz = abslx::UTCTimeZone();
  abslx::Time t = abslx::FromTimeT(0);

  TestFormatSpecifier(t, tz, "%a", "Thu");
  TestFormatSpecifier(t, tz, "%A", "Thursday");
  TestFormatSpecifier(t, tz, "%b", "Jan");
  TestFormatSpecifier(t, tz, "%B", "January");

  // %c should at least produce the numeric year and time-of-day.
  const std::string s =
      abslx::FormatTime("%c", abslx::FromTimeT(0), abslx::UTCTimeZone());
  EXPECT_THAT(s, HasSubstr("1970"));
  EXPECT_THAT(s, HasSubstr("00:00:00"));

  TestFormatSpecifier(t, tz, "%p", "AM");
  TestFormatSpecifier(t, tz, "%x", "01/01/70");
  TestFormatSpecifier(t, tz, "%X", "00:00:00");
}

TEST(FormatTime, ExtendedSeconds) {
  const abslx::TimeZone tz = abslx::UTCTimeZone();

  // No subseconds.
  abslx::Time t = abslx::FromTimeT(0) + abslx::Seconds(5);
  EXPECT_EQ("05", abslx::FormatTime("%E*S", t, tz));
  EXPECT_EQ("05.000000000000000", abslx::FormatTime("%E15S", t, tz));

  // With subseconds.
  t += abslx::Milliseconds(6) + abslx::Microseconds(7) + abslx::Nanoseconds(8);
  EXPECT_EQ("05.006007008", abslx::FormatTime("%E*S", t, tz));
  EXPECT_EQ("05", abslx::FormatTime("%E0S", t, tz));
  EXPECT_EQ("05.006007008000000", abslx::FormatTime("%E15S", t, tz));

  // Times before the Unix epoch.
  t = abslx::FromUnixMicros(-1);
  EXPECT_EQ("1969-12-31 23:59:59.999999",
            abslx::FormatTime("%Y-%m-%d %H:%M:%E*S", t, tz));

  // Here is a "%E*S" case we got wrong for a while.  While the first
  // instant below is correctly rendered as "...:07.333304", the second
  // one used to appear as "...:07.33330499999999999".
  t = abslx::FromUnixMicros(1395024427333304);
  EXPECT_EQ("2014-03-17 02:47:07.333304",
            abslx::FormatTime("%Y-%m-%d %H:%M:%E*S", t, tz));
  t += abslx::Microseconds(1);
  EXPECT_EQ("2014-03-17 02:47:07.333305",
            abslx::FormatTime("%Y-%m-%d %H:%M:%E*S", t, tz));
}

TEST(FormatTime, RFC1123FormatPadsYear) {  // locale specific
  abslx::TimeZone tz = abslx::UTCTimeZone();

  // A year of 77 should be padded to 0077.
  abslx::Time t = abslx::FromCivil(abslx::CivilSecond(77, 6, 28, 9, 8, 7), tz);
  EXPECT_EQ("Mon, 28 Jun 0077 09:08:07 +0000",
            abslx::FormatTime(abslx::RFC1123_full, t, tz));
  EXPECT_EQ("28 Jun 0077 09:08:07 +0000",
            abslx::FormatTime(abslx::RFC1123_no_wday, t, tz));
}

TEST(FormatTime, InfiniteTime) {
  abslx::TimeZone tz = abslx::time_internal::LoadTimeZone("America/Los_Angeles");

  // The format and timezone are ignored.
  EXPECT_EQ("infinite-future",
            abslx::FormatTime("%H:%M blah", abslx::InfiniteFuture(), tz));
  EXPECT_EQ("infinite-past",
            abslx::FormatTime("%H:%M blah", abslx::InfinitePast(), tz));
}

//
// Testing ParseTime()
//

TEST(ParseTime, Basics) {
  abslx::Time t = abslx::FromTimeT(1234567890);
  std::string err;

  // Simple edge cases.
  EXPECT_TRUE(abslx::ParseTime("", "", &t, &err)) << err;
  EXPECT_EQ(abslx::UnixEpoch(), t);  // everything defaulted
  EXPECT_TRUE(abslx::ParseTime(" ", " ", &t, &err)) << err;
  EXPECT_TRUE(abslx::ParseTime("  ", "  ", &t, &err)) << err;
  EXPECT_TRUE(abslx::ParseTime("x", "x", &t, &err)) << err;
  EXPECT_TRUE(abslx::ParseTime("xxx", "xxx", &t, &err)) << err;

  EXPECT_TRUE(abslx::ParseTime("%Y-%m-%d %H:%M:%S %z",
                              "2013-06-28 19:08:09 -0800", &t, &err))
      << err;
  const auto ci = abslx::FixedTimeZone(-8 * 60 * 60).At(t);
  EXPECT_EQ(abslx::CivilSecond(2013, 6, 28, 19, 8, 9), ci.cs);
  EXPECT_EQ(abslx::ZeroDuration(), ci.subsecond);
}

TEST(ParseTime, NullErrorString) {
  abslx::Time t;
  EXPECT_FALSE(abslx::ParseTime("%Q", "invalid format", &t, nullptr));
  EXPECT_FALSE(abslx::ParseTime("%H", "12 trailing data", &t, nullptr));
  EXPECT_FALSE(
      abslx::ParseTime("%H out of range", "42 out of range", &t, nullptr));
}

TEST(ParseTime, WithTimeZone) {
  const abslx::TimeZone tz =
      abslx::time_internal::LoadTimeZone("America/Los_Angeles");
  abslx::Time t;
  std::string e;

  // We can parse a string without a UTC offset if we supply a timezone.
  EXPECT_TRUE(
      abslx::ParseTime("%Y-%m-%d %H:%M:%S", "2013-06-28 19:08:09", tz, &t, &e))
      << e;
  auto ci = tz.At(t);
  EXPECT_EQ(abslx::CivilSecond(2013, 6, 28, 19, 8, 9), ci.cs);
  EXPECT_EQ(abslx::ZeroDuration(), ci.subsecond);

  // But the timezone is ignored when a UTC offset is present.
  EXPECT_TRUE(abslx::ParseTime("%Y-%m-%d %H:%M:%S %z",
                              "2013-06-28 19:08:09 +0800", tz, &t, &e))
      << e;
  ci = abslx::FixedTimeZone(8 * 60 * 60).At(t);
  EXPECT_EQ(abslx::CivilSecond(2013, 6, 28, 19, 8, 9), ci.cs);
  EXPECT_EQ(abslx::ZeroDuration(), ci.subsecond);
}

TEST(ParseTime, ErrorCases) {
  abslx::Time t = abslx::FromTimeT(0);
  std::string err;

  EXPECT_FALSE(abslx::ParseTime("%S", "123", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Illegal trailing data"));

  // Can't parse an illegal format specifier.
  err.clear();
  EXPECT_FALSE(abslx::ParseTime("%Q", "x", &t, &err)) << err;
  // Exact contents of "err" are platform-dependent because of
  // differences in the strptime implementation between macOS and Linux.
  EXPECT_FALSE(err.empty());

  // Fails because of trailing, unparsed data "blah".
  EXPECT_FALSE(abslx::ParseTime("%m-%d", "2-3 blah", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Illegal trailing data"));

  // Feb 31 requires normalization.
  EXPECT_FALSE(abslx::ParseTime("%m-%d", "2-31", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Out-of-range"));

  // Check that we cannot have spaces in UTC offsets.
  EXPECT_TRUE(abslx::ParseTime("%z", "-0203", &t, &err)) << err;
  EXPECT_FALSE(abslx::ParseTime("%z", "- 2 3", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Failed to parse"));
  EXPECT_TRUE(abslx::ParseTime("%Ez", "-02:03", &t, &err)) << err;
  EXPECT_FALSE(abslx::ParseTime("%Ez", "- 2: 3", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Failed to parse"));

  // Check that we reject other malformed UTC offsets.
  EXPECT_FALSE(abslx::ParseTime("%Ez", "+-08:00", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Failed to parse"));
  EXPECT_FALSE(abslx::ParseTime("%Ez", "-+08:00", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Failed to parse"));

  // Check that we do not accept "-0" in fields that allow zero.
  EXPECT_FALSE(abslx::ParseTime("%Y", "-0", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Failed to parse"));
  EXPECT_FALSE(abslx::ParseTime("%E4Y", "-0", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Failed to parse"));
  EXPECT_FALSE(abslx::ParseTime("%H", "-0", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Failed to parse"));
  EXPECT_FALSE(abslx::ParseTime("%M", "-0", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Failed to parse"));
  EXPECT_FALSE(abslx::ParseTime("%S", "-0", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Failed to parse"));
  EXPECT_FALSE(abslx::ParseTime("%z", "+-000", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Failed to parse"));
  EXPECT_FALSE(abslx::ParseTime("%Ez", "+-0:00", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Failed to parse"));
  EXPECT_FALSE(abslx::ParseTime("%z", "-00-0", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Illegal trailing data"));
  EXPECT_FALSE(abslx::ParseTime("%Ez", "-00:-0", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Illegal trailing data"));
}

TEST(ParseTime, ExtendedSeconds) {
  std::string err;
  abslx::Time t;

  // Here is a "%E*S" case we got wrong for a while.  The fractional
  // part of the first instant is less than 2^31 and was correctly
  // parsed, while the second (and any subsecond field >=2^31) failed.
  t = abslx::UnixEpoch();
  EXPECT_TRUE(abslx::ParseTime("%E*S", "0.2147483647", &t, &err)) << err;
  EXPECT_EQ(abslx::UnixEpoch() + abslx::Nanoseconds(214748364) +
                abslx::Nanoseconds(1) / 2,
            t);
  t = abslx::UnixEpoch();
  EXPECT_TRUE(abslx::ParseTime("%E*S", "0.2147483648", &t, &err)) << err;
  EXPECT_EQ(abslx::UnixEpoch() + abslx::Nanoseconds(214748364) +
                abslx::Nanoseconds(3) / 4,
            t);

  // We should also be able to specify long strings of digits far
  // beyond the current resolution and have them convert the same way.
  t = abslx::UnixEpoch();
  EXPECT_TRUE(abslx::ParseTime(
      "%E*S", "0.214748364801234567890123456789012345678901234567890123456789",
      &t, &err))
      << err;
  EXPECT_EQ(abslx::UnixEpoch() + abslx::Nanoseconds(214748364) +
                abslx::Nanoseconds(3) / 4,
            t);
}

TEST(ParseTime, ExtendedOffsetErrors) {
  std::string err;
  abslx::Time t;

  // %z against +-HHMM.
  EXPECT_FALSE(abslx::ParseTime("%z", "-123", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Illegal trailing data"));

  // %z against +-HH.
  EXPECT_FALSE(abslx::ParseTime("%z", "-1", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Failed to parse"));

  // %Ez against +-HH:MM.
  EXPECT_FALSE(abslx::ParseTime("%Ez", "-12:3", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Illegal trailing data"));

  // %Ez against +-HHMM.
  EXPECT_FALSE(abslx::ParseTime("%Ez", "-123", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Illegal trailing data"));

  // %Ez against +-HH.
  EXPECT_FALSE(abslx::ParseTime("%Ez", "-1", &t, &err)) << err;
  EXPECT_THAT(err, HasSubstr("Failed to parse"));
}

TEST(ParseTime, InfiniteTime) {
  abslx::Time t;
  std::string err;
  EXPECT_TRUE(abslx::ParseTime("%H:%M blah", "infinite-future", &t, &err));
  EXPECT_EQ(abslx::InfiniteFuture(), t);

  // Surrounding whitespace.
  EXPECT_TRUE(abslx::ParseTime("%H:%M blah", "  infinite-future", &t, &err));
  EXPECT_EQ(abslx::InfiniteFuture(), t);
  EXPECT_TRUE(abslx::ParseTime("%H:%M blah", "infinite-future  ", &t, &err));
  EXPECT_EQ(abslx::InfiniteFuture(), t);
  EXPECT_TRUE(abslx::ParseTime("%H:%M blah", "  infinite-future  ", &t, &err));
  EXPECT_EQ(abslx::InfiniteFuture(), t);

  EXPECT_TRUE(abslx::ParseTime("%H:%M blah", "infinite-past", &t, &err));
  EXPECT_EQ(abslx::InfinitePast(), t);

  // Surrounding whitespace.
  EXPECT_TRUE(abslx::ParseTime("%H:%M blah", "  infinite-past", &t, &err));
  EXPECT_EQ(abslx::InfinitePast(), t);
  EXPECT_TRUE(abslx::ParseTime("%H:%M blah", "infinite-past  ", &t, &err));
  EXPECT_EQ(abslx::InfinitePast(), t);
  EXPECT_TRUE(abslx::ParseTime("%H:%M blah", "  infinite-past  ", &t, &err));
  EXPECT_EQ(abslx::InfinitePast(), t);

  // "infinite-future" as literal string
  abslx::TimeZone tz = abslx::UTCTimeZone();
  EXPECT_TRUE(abslx::ParseTime("infinite-future %H:%M", "infinite-future 03:04",
                              &t, &err));
  EXPECT_NE(abslx::InfiniteFuture(), t);
  EXPECT_EQ(3, tz.At(t).cs.hour());
  EXPECT_EQ(4, tz.At(t).cs.minute());

  // "infinite-past" as literal string
  EXPECT_TRUE(
      abslx::ParseTime("infinite-past %H:%M", "infinite-past 03:04", &t, &err));
  EXPECT_NE(abslx::InfinitePast(), t);
  EXPECT_EQ(3, tz.At(t).cs.hour());
  EXPECT_EQ(4, tz.At(t).cs.minute());

  // The input doesn't match the format.
  EXPECT_FALSE(abslx::ParseTime("infinite-future %H:%M", "03:04", &t, &err));
  EXPECT_FALSE(abslx::ParseTime("infinite-past %H:%M", "03:04", &t, &err));
}

TEST(ParseTime, FailsOnUnrepresentableTime) {
  const abslx::TimeZone utc = abslx::UTCTimeZone();
  abslx::Time t;
  EXPECT_FALSE(
      abslx::ParseTime("%Y-%m-%d", "-292277022657-01-27", utc, &t, nullptr));
  EXPECT_TRUE(
      abslx::ParseTime("%Y-%m-%d", "-292277022657-01-28", utc, &t, nullptr));
  EXPECT_TRUE(
      abslx::ParseTime("%Y-%m-%d", "292277026596-12-04", utc, &t, nullptr));
  EXPECT_FALSE(
      abslx::ParseTime("%Y-%m-%d", "292277026596-12-05", utc, &t, nullptr));
}

//
// Roundtrip test for FormatTime()/ParseTime().
//

TEST(FormatParse, RoundTrip) {
  const abslx::TimeZone lax =
      abslx::time_internal::LoadTimeZone("America/Los_Angeles");
  const abslx::Time in =
      abslx::FromCivil(abslx::CivilSecond(1977, 6, 28, 9, 8, 7), lax);
  const abslx::Duration subseconds = abslx::Nanoseconds(654321);
  std::string err;

  // RFC3339, which renders subseconds.
  {
    abslx::Time out;
    const std::string s =
        abslx::FormatTime(abslx::RFC3339_full, in + subseconds, lax);
    EXPECT_TRUE(abslx::ParseTime(abslx::RFC3339_full, s, &out, &err))
        << s << ": " << err;
    EXPECT_EQ(in + subseconds, out);  // RFC3339_full includes %Ez
  }

  // RFC1123, which only does whole seconds.
  {
    abslx::Time out;
    const std::string s = abslx::FormatTime(abslx::RFC1123_full, in, lax);
    EXPECT_TRUE(abslx::ParseTime(abslx::RFC1123_full, s, &out, &err))
        << s << ": " << err;
    EXPECT_EQ(in, out);  // RFC1123_full includes %z
  }

  // `abslx::FormatTime()` falls back to strftime() for "%c", which appears to
  // work. On Windows, `abslx::ParseTime()` falls back to std::get_time() which
  // appears to fail on "%c" (or at least on the "%c" text produced by
  // `strftime()`). This makes it fail the round-trip test.
  //
  // Under the emscripten compiler `abslx::ParseTime() falls back to
  // `strptime()`, but that ends up using a different definition for "%c"
  // compared to `strftime()`, also causing the round-trip test to fail
  // (see https://github.com/kripken/emscripten/pull/7491).
#if !defined(_MSC_VER) && !defined(__EMSCRIPTEN__)
  // Even though we don't know what %c will produce, it should roundtrip,
  // but only in the 0-offset timezone.
  {
    abslx::Time out;
    const std::string s = abslx::FormatTime("%c", in, abslx::UTCTimeZone());
    EXPECT_TRUE(abslx::ParseTime("%c", s, &out, &err)) << s << ": " << err;
    EXPECT_EQ(in, out);
  }
#endif  // !_MSC_VER && !__EMSCRIPTEN__
}

TEST(FormatParse, RoundTripDistantFuture) {
  const abslx::TimeZone tz = abslx::UTCTimeZone();
  const abslx::Time in =
      abslx::FromUnixSeconds(std::numeric_limits<int64_t>::max());
  std::string err;

  abslx::Time out;
  const std::string s = abslx::FormatTime(abslx::RFC3339_full, in, tz);
  EXPECT_TRUE(abslx::ParseTime(abslx::RFC3339_full, s, &out, &err))
      << s << ": " << err;
  EXPECT_EQ(in, out);
}

TEST(FormatParse, RoundTripDistantPast) {
  const abslx::TimeZone tz = abslx::UTCTimeZone();
  const abslx::Time in =
      abslx::FromUnixSeconds(std::numeric_limits<int64_t>::min());
  std::string err;

  abslx::Time out;
  const std::string s = abslx::FormatTime(abslx::RFC3339_full, in, tz);
  EXPECT_TRUE(abslx::ParseTime(abslx::RFC3339_full, s, &out, &err))
      << s << ": " << err;
  EXPECT_EQ(in, out);
}

}  // namespace
