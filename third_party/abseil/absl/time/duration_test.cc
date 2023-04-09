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

#if defined(_MSC_VER)
#include <winsock2.h>  // for timeval
#endif

#include <chrono>  // NOLINT(build/c++11)
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <limits>
#include <random>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/time/time.h"

namespace {

constexpr int64_t kint64max = std::numeric_limits<int64_t>::max();
constexpr int64_t kint64min = std::numeric_limits<int64_t>::min();

// Approximates the given number of years. This is only used to make some test
// code more readable.
abslx::Duration ApproxYears(int64_t n) { return abslx::Hours(n) * 365 * 24; }

// A gMock matcher to match timespec values. Use this matcher like:
// timespec ts1, ts2;
// EXPECT_THAT(ts1, TimespecMatcher(ts2));
MATCHER_P(TimespecMatcher, ts, "") {
  if (ts.tv_sec == arg.tv_sec && ts.tv_nsec == arg.tv_nsec)
    return true;
  *result_listener << "expected: {" << ts.tv_sec << ", " << ts.tv_nsec << "} ";
  *result_listener << "actual: {" << arg.tv_sec << ", " << arg.tv_nsec << "}";
  return false;
}

// A gMock matcher to match timeval values. Use this matcher like:
// timeval tv1, tv2;
// EXPECT_THAT(tv1, TimevalMatcher(tv2));
MATCHER_P(TimevalMatcher, tv, "") {
  if (tv.tv_sec == arg.tv_sec && tv.tv_usec == arg.tv_usec)
    return true;
  *result_listener << "expected: {" << tv.tv_sec << ", " << tv.tv_usec << "} ";
  *result_listener << "actual: {" << arg.tv_sec << ", " << arg.tv_usec << "}";
  return false;
}

TEST(Duration, ConstExpr) {
  constexpr abslx::Duration d0 = abslx::ZeroDuration();
  static_assert(d0 == abslx::ZeroDuration(), "ZeroDuration()");
  constexpr abslx::Duration d1 = abslx::Seconds(1);
  static_assert(d1 == abslx::Seconds(1), "Seconds(1)");
  static_assert(d1 != abslx::ZeroDuration(), "Seconds(1)");
  constexpr abslx::Duration d2 = abslx::InfiniteDuration();
  static_assert(d2 == abslx::InfiniteDuration(), "InfiniteDuration()");
  static_assert(d2 != abslx::ZeroDuration(), "InfiniteDuration()");
}

TEST(Duration, ValueSemantics) {
  // If this compiles, the test passes.
  constexpr abslx::Duration a;      // Default construction
  constexpr abslx::Duration b = a;  // Copy construction
  constexpr abslx::Duration c(b);   // Copy construction (again)

  abslx::Duration d;
  d = c;  // Assignment
}

TEST(Duration, Factories) {
  constexpr abslx::Duration zero = abslx::ZeroDuration();
  constexpr abslx::Duration nano = abslx::Nanoseconds(1);
  constexpr abslx::Duration micro = abslx::Microseconds(1);
  constexpr abslx::Duration milli = abslx::Milliseconds(1);
  constexpr abslx::Duration sec = abslx::Seconds(1);
  constexpr abslx::Duration min = abslx::Minutes(1);
  constexpr abslx::Duration hour = abslx::Hours(1);

  EXPECT_EQ(zero, abslx::Duration());
  EXPECT_EQ(zero, abslx::Seconds(0));
  EXPECT_EQ(nano, abslx::Nanoseconds(1));
  EXPECT_EQ(micro, abslx::Nanoseconds(1000));
  EXPECT_EQ(milli, abslx::Microseconds(1000));
  EXPECT_EQ(sec, abslx::Milliseconds(1000));
  EXPECT_EQ(min, abslx::Seconds(60));
  EXPECT_EQ(hour, abslx::Minutes(60));

  // Tests factory limits
  const abslx::Duration inf = abslx::InfiniteDuration();

  EXPECT_GT(inf, abslx::Seconds(kint64max));
  EXPECT_LT(-inf, abslx::Seconds(kint64min));
  EXPECT_LT(-inf, abslx::Seconds(-kint64max));

  EXPECT_EQ(inf, abslx::Minutes(kint64max));
  EXPECT_EQ(-inf, abslx::Minutes(kint64min));
  EXPECT_EQ(-inf, abslx::Minutes(-kint64max));
  EXPECT_GT(inf, abslx::Minutes(kint64max / 60));
  EXPECT_LT(-inf, abslx::Minutes(kint64min / 60));
  EXPECT_LT(-inf, abslx::Minutes(-kint64max / 60));

  EXPECT_EQ(inf, abslx::Hours(kint64max));
  EXPECT_EQ(-inf, abslx::Hours(kint64min));
  EXPECT_EQ(-inf, abslx::Hours(-kint64max));
  EXPECT_GT(inf, abslx::Hours(kint64max / 3600));
  EXPECT_LT(-inf, abslx::Hours(kint64min / 3600));
  EXPECT_LT(-inf, abslx::Hours(-kint64max / 3600));
}

TEST(Duration, ToConversion) {
#define TEST_DURATION_CONVERSION(UNIT)                                  \
  do {                                                                  \
    const abslx::Duration d = abslx::UNIT(1.5);                           \
    constexpr abslx::Duration z = abslx::ZeroDuration();                  \
    constexpr abslx::Duration inf = abslx::InfiniteDuration();            \
    constexpr double dbl_inf = std::numeric_limits<double>::infinity(); \
    EXPECT_EQ(kint64min, abslx::ToInt64##UNIT(-inf));                    \
    EXPECT_EQ(-1, abslx::ToInt64##UNIT(-d));                             \
    EXPECT_EQ(0, abslx::ToInt64##UNIT(z));                               \
    EXPECT_EQ(1, abslx::ToInt64##UNIT(d));                               \
    EXPECT_EQ(kint64max, abslx::ToInt64##UNIT(inf));                     \
    EXPECT_EQ(-dbl_inf, abslx::ToDouble##UNIT(-inf));                    \
    EXPECT_EQ(-1.5, abslx::ToDouble##UNIT(-d));                          \
    EXPECT_EQ(0, abslx::ToDouble##UNIT(z));                              \
    EXPECT_EQ(1.5, abslx::ToDouble##UNIT(d));                            \
    EXPECT_EQ(dbl_inf, abslx::ToDouble##UNIT(inf));                      \
  } while (0)

  TEST_DURATION_CONVERSION(Nanoseconds);
  TEST_DURATION_CONVERSION(Microseconds);
  TEST_DURATION_CONVERSION(Milliseconds);
  TEST_DURATION_CONVERSION(Seconds);
  TEST_DURATION_CONVERSION(Minutes);
  TEST_DURATION_CONVERSION(Hours);

#undef TEST_DURATION_CONVERSION
}

template <int64_t N>
void TestToConversion() {
  constexpr abslx::Duration nano = abslx::Nanoseconds(N);
  EXPECT_EQ(N, abslx::ToInt64Nanoseconds(nano));
  EXPECT_EQ(0, abslx::ToInt64Microseconds(nano));
  EXPECT_EQ(0, abslx::ToInt64Milliseconds(nano));
  EXPECT_EQ(0, abslx::ToInt64Seconds(nano));
  EXPECT_EQ(0, abslx::ToInt64Minutes(nano));
  EXPECT_EQ(0, abslx::ToInt64Hours(nano));
  const abslx::Duration micro = abslx::Microseconds(N);
  EXPECT_EQ(N * 1000, abslx::ToInt64Nanoseconds(micro));
  EXPECT_EQ(N, abslx::ToInt64Microseconds(micro));
  EXPECT_EQ(0, abslx::ToInt64Milliseconds(micro));
  EXPECT_EQ(0, abslx::ToInt64Seconds(micro));
  EXPECT_EQ(0, abslx::ToInt64Minutes(micro));
  EXPECT_EQ(0, abslx::ToInt64Hours(micro));
  const abslx::Duration milli = abslx::Milliseconds(N);
  EXPECT_EQ(N * 1000 * 1000, abslx::ToInt64Nanoseconds(milli));
  EXPECT_EQ(N * 1000, abslx::ToInt64Microseconds(milli));
  EXPECT_EQ(N, abslx::ToInt64Milliseconds(milli));
  EXPECT_EQ(0, abslx::ToInt64Seconds(milli));
  EXPECT_EQ(0, abslx::ToInt64Minutes(milli));
  EXPECT_EQ(0, abslx::ToInt64Hours(milli));
  const abslx::Duration sec = abslx::Seconds(N);
  EXPECT_EQ(N * 1000 * 1000 * 1000, abslx::ToInt64Nanoseconds(sec));
  EXPECT_EQ(N * 1000 * 1000, abslx::ToInt64Microseconds(sec));
  EXPECT_EQ(N * 1000, abslx::ToInt64Milliseconds(sec));
  EXPECT_EQ(N, abslx::ToInt64Seconds(sec));
  EXPECT_EQ(0, abslx::ToInt64Minutes(sec));
  EXPECT_EQ(0, abslx::ToInt64Hours(sec));
  const abslx::Duration min = abslx::Minutes(N);
  EXPECT_EQ(N * 60 * 1000 * 1000 * 1000, abslx::ToInt64Nanoseconds(min));
  EXPECT_EQ(N * 60 * 1000 * 1000, abslx::ToInt64Microseconds(min));
  EXPECT_EQ(N * 60 * 1000, abslx::ToInt64Milliseconds(min));
  EXPECT_EQ(N * 60, abslx::ToInt64Seconds(min));
  EXPECT_EQ(N, abslx::ToInt64Minutes(min));
  EXPECT_EQ(0, abslx::ToInt64Hours(min));
  const abslx::Duration hour = abslx::Hours(N);
  EXPECT_EQ(N * 60 * 60 * 1000 * 1000 * 1000, abslx::ToInt64Nanoseconds(hour));
  EXPECT_EQ(N * 60 * 60 * 1000 * 1000, abslx::ToInt64Microseconds(hour));
  EXPECT_EQ(N * 60 * 60 * 1000, abslx::ToInt64Milliseconds(hour));
  EXPECT_EQ(N * 60 * 60, abslx::ToInt64Seconds(hour));
  EXPECT_EQ(N * 60, abslx::ToInt64Minutes(hour));
  EXPECT_EQ(N, abslx::ToInt64Hours(hour));
}

TEST(Duration, ToConversionDeprecated) {
  TestToConversion<43>();
  TestToConversion<1>();
  TestToConversion<0>();
  TestToConversion<-1>();
  TestToConversion<-43>();
}

template <int64_t N>
void TestFromChronoBasicEquality() {
  using std::chrono::nanoseconds;
  using std::chrono::microseconds;
  using std::chrono::milliseconds;
  using std::chrono::seconds;
  using std::chrono::minutes;
  using std::chrono::hours;

  static_assert(abslx::Nanoseconds(N) == abslx::FromChrono(nanoseconds(N)), "");
  static_assert(abslx::Microseconds(N) == abslx::FromChrono(microseconds(N)), "");
  static_assert(abslx::Milliseconds(N) == abslx::FromChrono(milliseconds(N)), "");
  static_assert(abslx::Seconds(N) == abslx::FromChrono(seconds(N)), "");
  static_assert(abslx::Minutes(N) == abslx::FromChrono(minutes(N)), "");
  static_assert(abslx::Hours(N) == abslx::FromChrono(hours(N)), "");
}

TEST(Duration, FromChrono) {
  TestFromChronoBasicEquality<-123>();
  TestFromChronoBasicEquality<-1>();
  TestFromChronoBasicEquality<0>();
  TestFromChronoBasicEquality<1>();
  TestFromChronoBasicEquality<123>();

  // Minutes (might, depending on the platform) saturate at +inf.
  const auto chrono_minutes_max = std::chrono::minutes::max();
  const auto minutes_max = abslx::FromChrono(chrono_minutes_max);
  const int64_t minutes_max_count = chrono_minutes_max.count();
  if (minutes_max_count > kint64max / 60) {
    EXPECT_EQ(abslx::InfiniteDuration(), minutes_max);
  } else {
    EXPECT_EQ(abslx::Minutes(minutes_max_count), minutes_max);
  }

  // Minutes (might, depending on the platform) saturate at -inf.
  const auto chrono_minutes_min = std::chrono::minutes::min();
  const auto minutes_min = abslx::FromChrono(chrono_minutes_min);
  const int64_t minutes_min_count = chrono_minutes_min.count();
  if (minutes_min_count < kint64min / 60) {
    EXPECT_EQ(-abslx::InfiniteDuration(), minutes_min);
  } else {
    EXPECT_EQ(abslx::Minutes(minutes_min_count), minutes_min);
  }

  // Hours (might, depending on the platform) saturate at +inf.
  const auto chrono_hours_max = std::chrono::hours::max();
  const auto hours_max = abslx::FromChrono(chrono_hours_max);
  const int64_t hours_max_count = chrono_hours_max.count();
  if (hours_max_count > kint64max / 3600) {
    EXPECT_EQ(abslx::InfiniteDuration(), hours_max);
  } else {
    EXPECT_EQ(abslx::Hours(hours_max_count), hours_max);
  }

  // Hours (might, depending on the platform) saturate at -inf.
  const auto chrono_hours_min = std::chrono::hours::min();
  const auto hours_min = abslx::FromChrono(chrono_hours_min);
  const int64_t hours_min_count = chrono_hours_min.count();
  if (hours_min_count < kint64min / 3600) {
    EXPECT_EQ(-abslx::InfiniteDuration(), hours_min);
  } else {
    EXPECT_EQ(abslx::Hours(hours_min_count), hours_min);
  }
}

template <int64_t N>
void TestToChrono() {
  using std::chrono::nanoseconds;
  using std::chrono::microseconds;
  using std::chrono::milliseconds;
  using std::chrono::seconds;
  using std::chrono::minutes;
  using std::chrono::hours;

  EXPECT_EQ(nanoseconds(N), abslx::ToChronoNanoseconds(abslx::Nanoseconds(N)));
  EXPECT_EQ(microseconds(N), abslx::ToChronoMicroseconds(abslx::Microseconds(N)));
  EXPECT_EQ(milliseconds(N), abslx::ToChronoMilliseconds(abslx::Milliseconds(N)));
  EXPECT_EQ(seconds(N), abslx::ToChronoSeconds(abslx::Seconds(N)));

  constexpr auto absl_minutes = abslx::Minutes(N);
  auto chrono_minutes = minutes(N);
  if (absl_minutes == -abslx::InfiniteDuration()) {
    chrono_minutes = minutes::min();
  } else if (absl_minutes == abslx::InfiniteDuration()) {
    chrono_minutes = minutes::max();
  }
  EXPECT_EQ(chrono_minutes, abslx::ToChronoMinutes(absl_minutes));

  constexpr auto absl_hours = abslx::Hours(N);
  auto chrono_hours = hours(N);
  if (absl_hours == -abslx::InfiniteDuration()) {
    chrono_hours = hours::min();
  } else if (absl_hours == abslx::InfiniteDuration()) {
    chrono_hours = hours::max();
  }
  EXPECT_EQ(chrono_hours, abslx::ToChronoHours(absl_hours));
}

TEST(Duration, ToChrono) {
  using std::chrono::nanoseconds;
  using std::chrono::microseconds;
  using std::chrono::milliseconds;
  using std::chrono::seconds;
  using std::chrono::minutes;
  using std::chrono::hours;

  TestToChrono<kint64min>();
  TestToChrono<-1>();
  TestToChrono<0>();
  TestToChrono<1>();
  TestToChrono<kint64max>();

  // Verify truncation toward zero.
  const auto tick = abslx::Nanoseconds(1) / 4;
  EXPECT_EQ(nanoseconds(0), abslx::ToChronoNanoseconds(tick));
  EXPECT_EQ(nanoseconds(0), abslx::ToChronoNanoseconds(-tick));
  EXPECT_EQ(microseconds(0), abslx::ToChronoMicroseconds(tick));
  EXPECT_EQ(microseconds(0), abslx::ToChronoMicroseconds(-tick));
  EXPECT_EQ(milliseconds(0), abslx::ToChronoMilliseconds(tick));
  EXPECT_EQ(milliseconds(0), abslx::ToChronoMilliseconds(-tick));
  EXPECT_EQ(seconds(0), abslx::ToChronoSeconds(tick));
  EXPECT_EQ(seconds(0), abslx::ToChronoSeconds(-tick));
  EXPECT_EQ(minutes(0), abslx::ToChronoMinutes(tick));
  EXPECT_EQ(minutes(0), abslx::ToChronoMinutes(-tick));
  EXPECT_EQ(hours(0), abslx::ToChronoHours(tick));
  EXPECT_EQ(hours(0), abslx::ToChronoHours(-tick));

  // Verifies +/- infinity saturation at max/min.
  constexpr auto inf = abslx::InfiniteDuration();
  EXPECT_EQ(nanoseconds::min(), abslx::ToChronoNanoseconds(-inf));
  EXPECT_EQ(nanoseconds::max(), abslx::ToChronoNanoseconds(inf));
  EXPECT_EQ(microseconds::min(), abslx::ToChronoMicroseconds(-inf));
  EXPECT_EQ(microseconds::max(), abslx::ToChronoMicroseconds(inf));
  EXPECT_EQ(milliseconds::min(), abslx::ToChronoMilliseconds(-inf));
  EXPECT_EQ(milliseconds::max(), abslx::ToChronoMilliseconds(inf));
  EXPECT_EQ(seconds::min(), abslx::ToChronoSeconds(-inf));
  EXPECT_EQ(seconds::max(), abslx::ToChronoSeconds(inf));
  EXPECT_EQ(minutes::min(), abslx::ToChronoMinutes(-inf));
  EXPECT_EQ(minutes::max(), abslx::ToChronoMinutes(inf));
  EXPECT_EQ(hours::min(), abslx::ToChronoHours(-inf));
  EXPECT_EQ(hours::max(), abslx::ToChronoHours(inf));
}

TEST(Duration, FactoryOverloads) {
  enum E { kOne = 1 };
#define TEST_FACTORY_OVERLOADS(NAME)                                          \
  EXPECT_EQ(1, NAME(kOne) / NAME(kOne));                                      \
  EXPECT_EQ(1, NAME(static_cast<int8_t>(1)) / NAME(1));                       \
  EXPECT_EQ(1, NAME(static_cast<int16_t>(1)) / NAME(1));                      \
  EXPECT_EQ(1, NAME(static_cast<int32_t>(1)) / NAME(1));                      \
  EXPECT_EQ(1, NAME(static_cast<int64_t>(1)) / NAME(1));                      \
  EXPECT_EQ(1, NAME(static_cast<uint8_t>(1)) / NAME(1));                      \
  EXPECT_EQ(1, NAME(static_cast<uint16_t>(1)) / NAME(1));                     \
  EXPECT_EQ(1, NAME(static_cast<uint32_t>(1)) / NAME(1));                     \
  EXPECT_EQ(1, NAME(static_cast<uint64_t>(1)) / NAME(1));                     \
  EXPECT_EQ(NAME(1) / 2, NAME(static_cast<float>(0.5)));                      \
  EXPECT_EQ(NAME(1) / 2, NAME(static_cast<double>(0.5)));                     \
  EXPECT_EQ(1.5, abslx::FDivDuration(NAME(static_cast<float>(1.5)), NAME(1))); \
  EXPECT_EQ(1.5, abslx::FDivDuration(NAME(static_cast<double>(1.5)), NAME(1)));

  TEST_FACTORY_OVERLOADS(abslx::Nanoseconds);
  TEST_FACTORY_OVERLOADS(abslx::Microseconds);
  TEST_FACTORY_OVERLOADS(abslx::Milliseconds);
  TEST_FACTORY_OVERLOADS(abslx::Seconds);
  TEST_FACTORY_OVERLOADS(abslx::Minutes);
  TEST_FACTORY_OVERLOADS(abslx::Hours);

#undef TEST_FACTORY_OVERLOADS

  EXPECT_EQ(abslx::Milliseconds(1500), abslx::Seconds(1.5));
  EXPECT_LT(abslx::Nanoseconds(1), abslx::Nanoseconds(1.5));
  EXPECT_GT(abslx::Nanoseconds(2), abslx::Nanoseconds(1.5));

  const double dbl_inf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(abslx::InfiniteDuration(), abslx::Nanoseconds(dbl_inf));
  EXPECT_EQ(abslx::InfiniteDuration(), abslx::Microseconds(dbl_inf));
  EXPECT_EQ(abslx::InfiniteDuration(), abslx::Milliseconds(dbl_inf));
  EXPECT_EQ(abslx::InfiniteDuration(), abslx::Seconds(dbl_inf));
  EXPECT_EQ(abslx::InfiniteDuration(), abslx::Minutes(dbl_inf));
  EXPECT_EQ(abslx::InfiniteDuration(), abslx::Hours(dbl_inf));
  EXPECT_EQ(-abslx::InfiniteDuration(), abslx::Nanoseconds(-dbl_inf));
  EXPECT_EQ(-abslx::InfiniteDuration(), abslx::Microseconds(-dbl_inf));
  EXPECT_EQ(-abslx::InfiniteDuration(), abslx::Milliseconds(-dbl_inf));
  EXPECT_EQ(-abslx::InfiniteDuration(), abslx::Seconds(-dbl_inf));
  EXPECT_EQ(-abslx::InfiniteDuration(), abslx::Minutes(-dbl_inf));
  EXPECT_EQ(-abslx::InfiniteDuration(), abslx::Hours(-dbl_inf));
}

TEST(Duration, InfinityExamples) {
  // These examples are used in the documentation in time.h. They are
  // written so that they can be copy-n-pasted easily.

  constexpr abslx::Duration inf = abslx::InfiniteDuration();
  constexpr abslx::Duration d = abslx::Seconds(1);  // Any finite duration

  EXPECT_TRUE(inf == inf + inf);
  EXPECT_TRUE(inf == inf + d);
  EXPECT_TRUE(inf == inf - inf);
  EXPECT_TRUE(-inf == d - inf);

  EXPECT_TRUE(inf == d * 1e100);
  EXPECT_TRUE(0 == d / inf);  // NOLINT(readability/check)

  // Division by zero returns infinity, or kint64min/MAX where necessary.
  EXPECT_TRUE(inf == d / 0);
  EXPECT_TRUE(kint64max == d / abslx::ZeroDuration());
}

TEST(Duration, InfinityComparison) {
  const abslx::Duration inf = abslx::InfiniteDuration();
  const abslx::Duration any_dur = abslx::Seconds(1);

  // Equality
  EXPECT_EQ(inf, inf);
  EXPECT_EQ(-inf, -inf);
  EXPECT_NE(inf, -inf);
  EXPECT_NE(any_dur, inf);
  EXPECT_NE(any_dur, -inf);

  // Relational
  EXPECT_GT(inf, any_dur);
  EXPECT_LT(-inf, any_dur);
  EXPECT_LT(-inf, inf);
  EXPECT_GT(inf, -inf);
}

TEST(Duration, InfinityAddition) {
  const abslx::Duration sec_max = abslx::Seconds(kint64max);
  const abslx::Duration sec_min = abslx::Seconds(kint64min);
  const abslx::Duration any_dur = abslx::Seconds(1);
  const abslx::Duration inf = abslx::InfiniteDuration();

  // Addition
  EXPECT_EQ(inf, inf + inf);
  EXPECT_EQ(inf, inf + -inf);
  EXPECT_EQ(-inf, -inf + inf);
  EXPECT_EQ(-inf, -inf + -inf);

  EXPECT_EQ(inf, inf + any_dur);
  EXPECT_EQ(inf, any_dur + inf);
  EXPECT_EQ(-inf, -inf + any_dur);
  EXPECT_EQ(-inf, any_dur + -inf);

  // Interesting case
  abslx::Duration almost_inf = sec_max + abslx::Nanoseconds(999999999);
  EXPECT_GT(inf, almost_inf);
  almost_inf += -abslx::Nanoseconds(999999999);
  EXPECT_GT(inf, almost_inf);

  // Addition overflow/underflow
  EXPECT_EQ(inf, sec_max + abslx::Seconds(1));
  EXPECT_EQ(inf, sec_max + sec_max);
  EXPECT_EQ(-inf, sec_min + -abslx::Seconds(1));
  EXPECT_EQ(-inf, sec_min + -sec_max);

  // For reference: IEEE 754 behavior
  const double dbl_inf = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(std::isinf(dbl_inf + dbl_inf));
  EXPECT_TRUE(std::isnan(dbl_inf + -dbl_inf));  // We return inf
  EXPECT_TRUE(std::isnan(-dbl_inf + dbl_inf));  // We return inf
  EXPECT_TRUE(std::isinf(-dbl_inf + -dbl_inf));
}

TEST(Duration, InfinitySubtraction) {
  const abslx::Duration sec_max = abslx::Seconds(kint64max);
  const abslx::Duration sec_min = abslx::Seconds(kint64min);
  const abslx::Duration any_dur = abslx::Seconds(1);
  const abslx::Duration inf = abslx::InfiniteDuration();

  // Subtraction
  EXPECT_EQ(inf, inf - inf);
  EXPECT_EQ(inf, inf - -inf);
  EXPECT_EQ(-inf, -inf - inf);
  EXPECT_EQ(-inf, -inf - -inf);

  EXPECT_EQ(inf, inf - any_dur);
  EXPECT_EQ(-inf, any_dur - inf);
  EXPECT_EQ(-inf, -inf - any_dur);
  EXPECT_EQ(inf, any_dur - -inf);

  // Subtraction overflow/underflow
  EXPECT_EQ(inf, sec_max - -abslx::Seconds(1));
  EXPECT_EQ(inf, sec_max - -sec_max);
  EXPECT_EQ(-inf, sec_min - abslx::Seconds(1));
  EXPECT_EQ(-inf, sec_min - sec_max);

  // Interesting case
  abslx::Duration almost_neg_inf = sec_min;
  EXPECT_LT(-inf, almost_neg_inf);
  almost_neg_inf -= -abslx::Nanoseconds(1);
  EXPECT_LT(-inf, almost_neg_inf);

  // For reference: IEEE 754 behavior
  const double dbl_inf = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(std::isnan(dbl_inf - dbl_inf));  // We return inf
  EXPECT_TRUE(std::isinf(dbl_inf - -dbl_inf));
  EXPECT_TRUE(std::isinf(-dbl_inf - dbl_inf));
  EXPECT_TRUE(std::isnan(-dbl_inf - -dbl_inf));  // We return inf
}

TEST(Duration, InfinityMultiplication) {
  const abslx::Duration sec_max = abslx::Seconds(kint64max);
  const abslx::Duration sec_min = abslx::Seconds(kint64min);
  const abslx::Duration inf = abslx::InfiniteDuration();

#define TEST_INF_MUL_WITH_TYPE(T)                                     \
  EXPECT_EQ(inf, inf * static_cast<T>(2));                            \
  EXPECT_EQ(-inf, inf * static_cast<T>(-2));                          \
  EXPECT_EQ(-inf, -inf * static_cast<T>(2));                          \
  EXPECT_EQ(inf, -inf * static_cast<T>(-2));                          \
  EXPECT_EQ(inf, inf * static_cast<T>(0));                            \
  EXPECT_EQ(-inf, -inf * static_cast<T>(0));                          \
  EXPECT_EQ(inf, sec_max * static_cast<T>(2));                        \
  EXPECT_EQ(inf, sec_min * static_cast<T>(-2));                       \
  EXPECT_EQ(inf, (sec_max / static_cast<T>(2)) * static_cast<T>(3));  \
  EXPECT_EQ(-inf, sec_max * static_cast<T>(-2));                      \
  EXPECT_EQ(-inf, sec_min * static_cast<T>(2));                       \
  EXPECT_EQ(-inf, (sec_min / static_cast<T>(2)) * static_cast<T>(3));

  TEST_INF_MUL_WITH_TYPE(int64_t);  // NOLINT(readability/function)
  TEST_INF_MUL_WITH_TYPE(double);   // NOLINT(readability/function)

#undef TEST_INF_MUL_WITH_TYPE

  const double dbl_inf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(inf, inf * dbl_inf);
  EXPECT_EQ(-inf, -inf * dbl_inf);
  EXPECT_EQ(-inf, inf * -dbl_inf);
  EXPECT_EQ(inf, -inf * -dbl_inf);

  const abslx::Duration any_dur = abslx::Seconds(1);
  EXPECT_EQ(inf, any_dur * dbl_inf);
  EXPECT_EQ(-inf, -any_dur * dbl_inf);
  EXPECT_EQ(-inf, any_dur * -dbl_inf);
  EXPECT_EQ(inf, -any_dur * -dbl_inf);

  // Fixed-point multiplication will produce a finite value, whereas floating
  // point fuzziness will overflow to inf.
  EXPECT_NE(abslx::InfiniteDuration(), abslx::Seconds(1) * kint64max);
  EXPECT_EQ(inf, abslx::Seconds(1) * static_cast<double>(kint64max));
  EXPECT_NE(-abslx::InfiniteDuration(), abslx::Seconds(1) * kint64min);
  EXPECT_EQ(-inf, abslx::Seconds(1) * static_cast<double>(kint64min));

  // Note that sec_max * or / by 1.0 overflows to inf due to the 53-bit
  // limitations of double.
  EXPECT_NE(inf, sec_max);
  EXPECT_NE(inf, sec_max / 1);
  EXPECT_EQ(inf, sec_max / 1.0);
  EXPECT_NE(inf, sec_max * 1);
  EXPECT_EQ(inf, sec_max * 1.0);
}

TEST(Duration, InfinityDivision) {
  const abslx::Duration sec_max = abslx::Seconds(kint64max);
  const abslx::Duration sec_min = abslx::Seconds(kint64min);
  const abslx::Duration inf = abslx::InfiniteDuration();

  // Division of Duration by a double
#define TEST_INF_DIV_WITH_TYPE(T)            \
  EXPECT_EQ(inf, inf / static_cast<T>(2));   \
  EXPECT_EQ(-inf, inf / static_cast<T>(-2)); \
  EXPECT_EQ(-inf, -inf / static_cast<T>(2)); \
  EXPECT_EQ(inf, -inf / static_cast<T>(-2));

  TEST_INF_DIV_WITH_TYPE(int64_t);  // NOLINT(readability/function)
  TEST_INF_DIV_WITH_TYPE(double);   // NOLINT(readability/function)

#undef TEST_INF_DIV_WITH_TYPE

  // Division of Duration by a double overflow/underflow
  EXPECT_EQ(inf, sec_max / 0.5);
  EXPECT_EQ(inf, sec_min / -0.5);
  EXPECT_EQ(inf, ((sec_max / 0.5) + abslx::Seconds(1)) / 0.5);
  EXPECT_EQ(-inf, sec_max / -0.5);
  EXPECT_EQ(-inf, sec_min / 0.5);
  EXPECT_EQ(-inf, ((sec_min / 0.5) - abslx::Seconds(1)) / 0.5);

  const double dbl_inf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(inf, inf / dbl_inf);
  EXPECT_EQ(-inf, inf / -dbl_inf);
  EXPECT_EQ(-inf, -inf / dbl_inf);
  EXPECT_EQ(inf, -inf / -dbl_inf);

  const abslx::Duration any_dur = abslx::Seconds(1);
  EXPECT_EQ(abslx::ZeroDuration(), any_dur / dbl_inf);
  EXPECT_EQ(abslx::ZeroDuration(), any_dur / -dbl_inf);
  EXPECT_EQ(abslx::ZeroDuration(), -any_dur / dbl_inf);
  EXPECT_EQ(abslx::ZeroDuration(), -any_dur / -dbl_inf);
}

TEST(Duration, InfinityModulus) {
  const abslx::Duration sec_max = abslx::Seconds(kint64max);
  const abslx::Duration any_dur = abslx::Seconds(1);
  const abslx::Duration inf = abslx::InfiniteDuration();

  EXPECT_EQ(inf, inf % inf);
  EXPECT_EQ(inf, inf % -inf);
  EXPECT_EQ(-inf, -inf % -inf);
  EXPECT_EQ(-inf, -inf % inf);

  EXPECT_EQ(any_dur, any_dur % inf);
  EXPECT_EQ(any_dur, any_dur % -inf);
  EXPECT_EQ(-any_dur, -any_dur % inf);
  EXPECT_EQ(-any_dur, -any_dur % -inf);

  EXPECT_EQ(inf, inf % -any_dur);
  EXPECT_EQ(inf, inf % any_dur);
  EXPECT_EQ(-inf, -inf % -any_dur);
  EXPECT_EQ(-inf, -inf % any_dur);

  // Remainder isn't affected by overflow.
  EXPECT_EQ(abslx::ZeroDuration(), sec_max % abslx::Seconds(1));
  EXPECT_EQ(abslx::ZeroDuration(), sec_max % abslx::Milliseconds(1));
  EXPECT_EQ(abslx::ZeroDuration(), sec_max % abslx::Microseconds(1));
  EXPECT_EQ(abslx::ZeroDuration(), sec_max % abslx::Nanoseconds(1));
  EXPECT_EQ(abslx::ZeroDuration(), sec_max % abslx::Nanoseconds(1) / 4);
}

TEST(Duration, InfinityIDiv) {
  const abslx::Duration sec_max = abslx::Seconds(kint64max);
  const abslx::Duration any_dur = abslx::Seconds(1);
  const abslx::Duration inf = abslx::InfiniteDuration();
  const double dbl_inf = std::numeric_limits<double>::infinity();

  // IDivDuration (int64_t return value + a remainer)
  abslx::Duration rem = abslx::ZeroDuration();
  EXPECT_EQ(kint64max, abslx::IDivDuration(inf, inf, &rem));
  EXPECT_EQ(inf, rem);

  rem = abslx::ZeroDuration();
  EXPECT_EQ(kint64max, abslx::IDivDuration(-inf, -inf, &rem));
  EXPECT_EQ(-inf, rem);

  rem = abslx::ZeroDuration();
  EXPECT_EQ(kint64max, abslx::IDivDuration(inf, any_dur, &rem));
  EXPECT_EQ(inf, rem);

  rem = abslx::ZeroDuration();
  EXPECT_EQ(0, abslx::IDivDuration(any_dur, inf, &rem));
  EXPECT_EQ(any_dur, rem);

  rem = abslx::ZeroDuration();
  EXPECT_EQ(kint64max, abslx::IDivDuration(-inf, -any_dur, &rem));
  EXPECT_EQ(-inf, rem);

  rem = abslx::ZeroDuration();
  EXPECT_EQ(0, abslx::IDivDuration(-any_dur, -inf, &rem));
  EXPECT_EQ(-any_dur, rem);

  rem = abslx::ZeroDuration();
  EXPECT_EQ(kint64min, abslx::IDivDuration(-inf, inf, &rem));
  EXPECT_EQ(-inf, rem);

  rem = abslx::ZeroDuration();
  EXPECT_EQ(kint64min, abslx::IDivDuration(inf, -inf, &rem));
  EXPECT_EQ(inf, rem);

  rem = abslx::ZeroDuration();
  EXPECT_EQ(kint64min, abslx::IDivDuration(-inf, any_dur, &rem));
  EXPECT_EQ(-inf, rem);

  rem = abslx::ZeroDuration();
  EXPECT_EQ(0, abslx::IDivDuration(-any_dur, inf, &rem));
  EXPECT_EQ(-any_dur, rem);

  rem = abslx::ZeroDuration();
  EXPECT_EQ(kint64min, abslx::IDivDuration(inf, -any_dur, &rem));
  EXPECT_EQ(inf, rem);

  rem = abslx::ZeroDuration();
  EXPECT_EQ(0, abslx::IDivDuration(any_dur, -inf, &rem));
  EXPECT_EQ(any_dur, rem);

  // IDivDuration overflow/underflow
  rem = any_dur;
  EXPECT_EQ(kint64max,
            abslx::IDivDuration(sec_max, abslx::Nanoseconds(1) / 4, &rem));
  EXPECT_EQ(sec_max - abslx::Nanoseconds(kint64max) / 4, rem);

  rem = any_dur;
  EXPECT_EQ(kint64max,
            abslx::IDivDuration(sec_max, abslx::Milliseconds(1), &rem));
  EXPECT_EQ(sec_max - abslx::Milliseconds(kint64max), rem);

  rem = any_dur;
  EXPECT_EQ(kint64max,
            abslx::IDivDuration(-sec_max, -abslx::Milliseconds(1), &rem));
  EXPECT_EQ(-sec_max + abslx::Milliseconds(kint64max), rem);

  rem = any_dur;
  EXPECT_EQ(kint64min,
            abslx::IDivDuration(-sec_max, abslx::Milliseconds(1), &rem));
  EXPECT_EQ(-sec_max - abslx::Milliseconds(kint64min), rem);

  rem = any_dur;
  EXPECT_EQ(kint64min,
            abslx::IDivDuration(sec_max, -abslx::Milliseconds(1), &rem));
  EXPECT_EQ(sec_max + abslx::Milliseconds(kint64min), rem);

  //
  // operator/(Duration, Duration) is a wrapper for IDivDuration().
  //

  // IEEE 754 says inf / inf should be nan, but int64_t doesn't have
  // nan so we'll return kint64max/kint64min instead.
  EXPECT_TRUE(std::isnan(dbl_inf / dbl_inf));
  EXPECT_EQ(kint64max, inf / inf);
  EXPECT_EQ(kint64max, -inf / -inf);
  EXPECT_EQ(kint64min, -inf / inf);
  EXPECT_EQ(kint64min, inf / -inf);

  EXPECT_TRUE(std::isinf(dbl_inf / 2.0));
  EXPECT_EQ(kint64max, inf / any_dur);
  EXPECT_EQ(kint64max, -inf / -any_dur);
  EXPECT_EQ(kint64min, -inf / any_dur);
  EXPECT_EQ(kint64min, inf / -any_dur);

  EXPECT_EQ(0.0, 2.0 / dbl_inf);
  EXPECT_EQ(0, any_dur / inf);
  EXPECT_EQ(0, any_dur / -inf);
  EXPECT_EQ(0, -any_dur / inf);
  EXPECT_EQ(0, -any_dur / -inf);
  EXPECT_EQ(0, abslx::ZeroDuration() / inf);

  // Division of Duration by a Duration overflow/underflow
  EXPECT_EQ(kint64max, sec_max / abslx::Milliseconds(1));
  EXPECT_EQ(kint64max, -sec_max / -abslx::Milliseconds(1));
  EXPECT_EQ(kint64min, -sec_max / abslx::Milliseconds(1));
  EXPECT_EQ(kint64min, sec_max / -abslx::Milliseconds(1));
}

TEST(Duration, InfinityFDiv) {
  const abslx::Duration any_dur = abslx::Seconds(1);
  const abslx::Duration inf = abslx::InfiniteDuration();
  const double dbl_inf = std::numeric_limits<double>::infinity();

  EXPECT_EQ(dbl_inf, abslx::FDivDuration(inf, inf));
  EXPECT_EQ(dbl_inf, abslx::FDivDuration(-inf, -inf));
  EXPECT_EQ(dbl_inf, abslx::FDivDuration(inf, any_dur));
  EXPECT_EQ(0.0, abslx::FDivDuration(any_dur, inf));
  EXPECT_EQ(dbl_inf, abslx::FDivDuration(-inf, -any_dur));
  EXPECT_EQ(0.0, abslx::FDivDuration(-any_dur, -inf));

  EXPECT_EQ(-dbl_inf, abslx::FDivDuration(-inf, inf));
  EXPECT_EQ(-dbl_inf, abslx::FDivDuration(inf, -inf));
  EXPECT_EQ(-dbl_inf, abslx::FDivDuration(-inf, any_dur));
  EXPECT_EQ(0.0, abslx::FDivDuration(-any_dur, inf));
  EXPECT_EQ(-dbl_inf, abslx::FDivDuration(inf, -any_dur));
  EXPECT_EQ(0.0, abslx::FDivDuration(any_dur, -inf));
}

TEST(Duration, DivisionByZero) {
  const abslx::Duration zero = abslx::ZeroDuration();
  const abslx::Duration inf = abslx::InfiniteDuration();
  const abslx::Duration any_dur = abslx::Seconds(1);
  const double dbl_inf = std::numeric_limits<double>::infinity();
  const double dbl_denorm = std::numeric_limits<double>::denorm_min();

  // Operator/(Duration, double)
  EXPECT_EQ(inf, zero / 0.0);
  EXPECT_EQ(-inf, zero / -0.0);
  EXPECT_EQ(inf, any_dur / 0.0);
  EXPECT_EQ(-inf, any_dur / -0.0);
  EXPECT_EQ(-inf, -any_dur / 0.0);
  EXPECT_EQ(inf, -any_dur / -0.0);

  // Tests dividing by a number very close to, but not quite zero.
  EXPECT_EQ(zero, zero / dbl_denorm);
  EXPECT_EQ(zero, zero / -dbl_denorm);
  EXPECT_EQ(inf, any_dur / dbl_denorm);
  EXPECT_EQ(-inf, any_dur / -dbl_denorm);
  EXPECT_EQ(-inf, -any_dur / dbl_denorm);
  EXPECT_EQ(inf, -any_dur / -dbl_denorm);

  // IDiv
  abslx::Duration rem = zero;
  EXPECT_EQ(kint64max, abslx::IDivDuration(zero, zero, &rem));
  EXPECT_EQ(inf, rem);

  rem = zero;
  EXPECT_EQ(kint64max, abslx::IDivDuration(any_dur, zero, &rem));
  EXPECT_EQ(inf, rem);

  rem = zero;
  EXPECT_EQ(kint64min, abslx::IDivDuration(-any_dur, zero, &rem));
  EXPECT_EQ(-inf, rem);

  // Operator/(Duration, Duration)
  EXPECT_EQ(kint64max, zero / zero);
  EXPECT_EQ(kint64max, any_dur / zero);
  EXPECT_EQ(kint64min, -any_dur / zero);

  // FDiv
  EXPECT_EQ(dbl_inf, abslx::FDivDuration(zero, zero));
  EXPECT_EQ(dbl_inf, abslx::FDivDuration(any_dur, zero));
  EXPECT_EQ(-dbl_inf, abslx::FDivDuration(-any_dur, zero));
}

TEST(Duration, NaN) {
  // Note that IEEE 754 does not define the behavior of a nan's sign when it is
  // copied, so the code below allows for either + or - InfiniteDuration.
#define TEST_NAN_HANDLING(NAME, NAN)           \
  do {                                         \
    const auto inf = abslx::InfiniteDuration(); \
    auto x = NAME(NAN);                        \
    EXPECT_TRUE(x == inf || x == -inf);        \
    auto y = NAME(42);                         \
    y *= NAN;                                  \
    EXPECT_TRUE(y == inf || y == -inf);        \
    auto z = NAME(42);                         \
    z /= NAN;                                  \
    EXPECT_TRUE(z == inf || z == -inf);        \
  } while (0)

  const double nan = std::numeric_limits<double>::quiet_NaN();
  TEST_NAN_HANDLING(abslx::Nanoseconds, nan);
  TEST_NAN_HANDLING(abslx::Microseconds, nan);
  TEST_NAN_HANDLING(abslx::Milliseconds, nan);
  TEST_NAN_HANDLING(abslx::Seconds, nan);
  TEST_NAN_HANDLING(abslx::Minutes, nan);
  TEST_NAN_HANDLING(abslx::Hours, nan);

  TEST_NAN_HANDLING(abslx::Nanoseconds, -nan);
  TEST_NAN_HANDLING(abslx::Microseconds, -nan);
  TEST_NAN_HANDLING(abslx::Milliseconds, -nan);
  TEST_NAN_HANDLING(abslx::Seconds, -nan);
  TEST_NAN_HANDLING(abslx::Minutes, -nan);
  TEST_NAN_HANDLING(abslx::Hours, -nan);

#undef TEST_NAN_HANDLING
}

TEST(Duration, Range) {
  const abslx::Duration range = ApproxYears(100 * 1e9);
  const abslx::Duration range_future = range;
  const abslx::Duration range_past = -range;

  EXPECT_LT(range_future, abslx::InfiniteDuration());
  EXPECT_GT(range_past, -abslx::InfiniteDuration());

  const abslx::Duration full_range = range_future - range_past;
  EXPECT_GT(full_range, abslx::ZeroDuration());
  EXPECT_LT(full_range, abslx::InfiniteDuration());

  const abslx::Duration neg_full_range = range_past - range_future;
  EXPECT_LT(neg_full_range, abslx::ZeroDuration());
  EXPECT_GT(neg_full_range, -abslx::InfiniteDuration());

  EXPECT_LT(neg_full_range, full_range);
  EXPECT_EQ(neg_full_range, -full_range);
}

TEST(Duration, RelationalOperators) {
#define TEST_REL_OPS(UNIT)               \
  static_assert(UNIT(2) == UNIT(2), ""); \
  static_assert(UNIT(1) != UNIT(2), ""); \
  static_assert(UNIT(1) < UNIT(2), "");  \
  static_assert(UNIT(3) > UNIT(2), "");  \
  static_assert(UNIT(1) <= UNIT(2), ""); \
  static_assert(UNIT(2) <= UNIT(2), ""); \
  static_assert(UNIT(3) >= UNIT(2), ""); \
  static_assert(UNIT(2) >= UNIT(2), "");

  TEST_REL_OPS(abslx::Nanoseconds);
  TEST_REL_OPS(abslx::Microseconds);
  TEST_REL_OPS(abslx::Milliseconds);
  TEST_REL_OPS(abslx::Seconds);
  TEST_REL_OPS(abslx::Minutes);
  TEST_REL_OPS(abslx::Hours);

#undef TEST_REL_OPS
}

TEST(Duration, Addition) {
#define TEST_ADD_OPS(UNIT)                  \
  do {                                      \
    EXPECT_EQ(UNIT(2), UNIT(1) + UNIT(1));  \
    EXPECT_EQ(UNIT(1), UNIT(2) - UNIT(1));  \
    EXPECT_EQ(UNIT(0), UNIT(2) - UNIT(2));  \
    EXPECT_EQ(UNIT(-1), UNIT(1) - UNIT(2)); \
    EXPECT_EQ(UNIT(-2), UNIT(0) - UNIT(2)); \
    EXPECT_EQ(UNIT(-2), UNIT(1) - UNIT(3)); \
    abslx::Duration a = UNIT(1);             \
    a += UNIT(1);                           \
    EXPECT_EQ(UNIT(2), a);                  \
    a -= UNIT(1);                           \
    EXPECT_EQ(UNIT(1), a);                  \
  } while (0)

  TEST_ADD_OPS(abslx::Nanoseconds);
  TEST_ADD_OPS(abslx::Microseconds);
  TEST_ADD_OPS(abslx::Milliseconds);
  TEST_ADD_OPS(abslx::Seconds);
  TEST_ADD_OPS(abslx::Minutes);
  TEST_ADD_OPS(abslx::Hours);

#undef TEST_ADD_OPS

  EXPECT_EQ(abslx::Seconds(2), abslx::Seconds(3) - 2 * abslx::Milliseconds(500));
  EXPECT_EQ(abslx::Seconds(2) + abslx::Milliseconds(500),
            abslx::Seconds(3) - abslx::Milliseconds(500));

  EXPECT_EQ(abslx::Seconds(1) + abslx::Milliseconds(998),
            abslx::Milliseconds(999) + abslx::Milliseconds(999));

  EXPECT_EQ(abslx::Milliseconds(-1),
            abslx::Milliseconds(998) - abslx::Milliseconds(999));

  // Tests fractions of a nanoseconds. These are implementation details only.
  EXPECT_GT(abslx::Nanoseconds(1), abslx::Nanoseconds(1) / 2);
  EXPECT_EQ(abslx::Nanoseconds(1),
            abslx::Nanoseconds(1) / 2 + abslx::Nanoseconds(1) / 2);
  EXPECT_GT(abslx::Nanoseconds(1) / 4, abslx::Nanoseconds(0));
  EXPECT_EQ(abslx::Nanoseconds(1) / 8, abslx::Nanoseconds(0));

  // Tests subtraction that will cause wrap around of the rep_lo_ bits.
  abslx::Duration d_7_5 = abslx::Seconds(7) + abslx::Milliseconds(500);
  abslx::Duration d_3_7 = abslx::Seconds(3) + abslx::Milliseconds(700);
  abslx::Duration ans_3_8 = abslx::Seconds(3) + abslx::Milliseconds(800);
  EXPECT_EQ(ans_3_8, d_7_5 - d_3_7);

  // Subtracting min_duration
  abslx::Duration min_dur = abslx::Seconds(kint64min);
  EXPECT_EQ(abslx::Seconds(0), min_dur - min_dur);
  EXPECT_EQ(abslx::Seconds(kint64max), abslx::Seconds(-1) - min_dur);
}

TEST(Duration, Negation) {
  // By storing negations of various values in constexpr variables we
  // verify that the initializers are constant expressions.
  constexpr abslx::Duration negated_zero_duration = -abslx::ZeroDuration();
  EXPECT_EQ(negated_zero_duration, abslx::ZeroDuration());

  constexpr abslx::Duration negated_infinite_duration =
      -abslx::InfiniteDuration();
  EXPECT_NE(negated_infinite_duration, abslx::InfiniteDuration());
  EXPECT_EQ(-negated_infinite_duration, abslx::InfiniteDuration());

  // The public APIs to check if a duration is infinite depend on using
  // -InfiniteDuration(), but we're trying to test operator- here, so we
  // need to use the lower-level internal query IsInfiniteDuration.
  EXPECT_TRUE(
      abslx::time_internal::IsInfiniteDuration(negated_infinite_duration));

  // The largest Duration is kint64max seconds and kTicksPerSecond - 1 ticks.
  // Using the abslx::time_internal::MakeDuration API is the cleanest way to
  // construct that Duration.
  constexpr abslx::Duration max_duration = abslx::time_internal::MakeDuration(
      kint64max, abslx::time_internal::kTicksPerSecond - 1);
  constexpr abslx::Duration negated_max_duration = -max_duration;
  // The largest negatable value is one tick above the minimum representable;
  // it's the negation of max_duration.
  constexpr abslx::Duration nearly_min_duration =
      abslx::time_internal::MakeDuration(kint64min, int64_t{1});
  constexpr abslx::Duration negated_nearly_min_duration = -nearly_min_duration;

  EXPECT_EQ(negated_max_duration, nearly_min_duration);
  EXPECT_EQ(negated_nearly_min_duration, max_duration);
  EXPECT_EQ(-(-max_duration), max_duration);

  constexpr abslx::Duration min_duration =
      abslx::time_internal::MakeDuration(kint64min);
  constexpr abslx::Duration negated_min_duration = -min_duration;
  EXPECT_EQ(negated_min_duration, abslx::InfiniteDuration());
}

TEST(Duration, AbsoluteValue) {
  EXPECT_EQ(abslx::ZeroDuration(), AbsDuration(abslx::ZeroDuration()));
  EXPECT_EQ(abslx::Seconds(1), AbsDuration(abslx::Seconds(1)));
  EXPECT_EQ(abslx::Seconds(1), AbsDuration(abslx::Seconds(-1)));

  EXPECT_EQ(abslx::InfiniteDuration(), AbsDuration(abslx::InfiniteDuration()));
  EXPECT_EQ(abslx::InfiniteDuration(), AbsDuration(-abslx::InfiniteDuration()));

  abslx::Duration max_dur =
      abslx::Seconds(kint64max) + (abslx::Seconds(1) - abslx::Nanoseconds(1) / 4);
  EXPECT_EQ(max_dur, AbsDuration(max_dur));

  abslx::Duration min_dur = abslx::Seconds(kint64min);
  EXPECT_EQ(abslx::InfiniteDuration(), AbsDuration(min_dur));
  EXPECT_EQ(max_dur, AbsDuration(min_dur + abslx::Nanoseconds(1) / 4));
}

TEST(Duration, Multiplication) {
#define TEST_MUL_OPS(UNIT)                                    \
  do {                                                        \
    EXPECT_EQ(UNIT(5), UNIT(2) * 2.5);                        \
    EXPECT_EQ(UNIT(2), UNIT(5) / 2.5);                        \
    EXPECT_EQ(UNIT(-5), UNIT(-2) * 2.5);                      \
    EXPECT_EQ(UNIT(-5), -UNIT(2) * 2.5);                      \
    EXPECT_EQ(UNIT(-5), UNIT(2) * -2.5);                      \
    EXPECT_EQ(UNIT(-2), UNIT(-5) / 2.5);                      \
    EXPECT_EQ(UNIT(-2), -UNIT(5) / 2.5);                      \
    EXPECT_EQ(UNIT(-2), UNIT(5) / -2.5);                      \
    EXPECT_EQ(UNIT(2), UNIT(11) % UNIT(3));                   \
    abslx::Duration a = UNIT(2);                               \
    a *= 2.5;                                                 \
    EXPECT_EQ(UNIT(5), a);                                    \
    a /= 2.5;                                                 \
    EXPECT_EQ(UNIT(2), a);                                    \
    a %= UNIT(1);                                             \
    EXPECT_EQ(UNIT(0), a);                                    \
    abslx::Duration big = UNIT(1000000000);                    \
    big *= 3;                                                 \
    big /= 3;                                                 \
    EXPECT_EQ(UNIT(1000000000), big);                         \
    EXPECT_EQ(-UNIT(2), -UNIT(2));                            \
    EXPECT_EQ(-UNIT(2), UNIT(2) * -1);                        \
    EXPECT_EQ(-UNIT(2), -1 * UNIT(2));                        \
    EXPECT_EQ(-UNIT(-2), UNIT(2));                            \
    EXPECT_EQ(2, UNIT(2) / UNIT(1));                          \
    abslx::Duration rem;                                       \
    EXPECT_EQ(2, abslx::IDivDuration(UNIT(2), UNIT(1), &rem)); \
    EXPECT_EQ(2.0, abslx::FDivDuration(UNIT(2), UNIT(1)));     \
  } while (0)

  TEST_MUL_OPS(abslx::Nanoseconds);
  TEST_MUL_OPS(abslx::Microseconds);
  TEST_MUL_OPS(abslx::Milliseconds);
  TEST_MUL_OPS(abslx::Seconds);
  TEST_MUL_OPS(abslx::Minutes);
  TEST_MUL_OPS(abslx::Hours);

#undef TEST_MUL_OPS

  // Ensures that multiplication and division by 1 with a maxed-out durations
  // doesn't lose precision.
  abslx::Duration max_dur =
      abslx::Seconds(kint64max) + (abslx::Seconds(1) - abslx::Nanoseconds(1) / 4);
  abslx::Duration min_dur = abslx::Seconds(kint64min);
  EXPECT_EQ(max_dur, max_dur * 1);
  EXPECT_EQ(max_dur, max_dur / 1);
  EXPECT_EQ(min_dur, min_dur * 1);
  EXPECT_EQ(min_dur, min_dur / 1);

  // Tests division on a Duration with a large number of significant digits.
  // Tests when the digits span hi and lo as well as only in hi.
  abslx::Duration sigfigs = abslx::Seconds(2000000000) + abslx::Nanoseconds(3);
  EXPECT_EQ(abslx::Seconds(666666666) + abslx::Nanoseconds(666666667) +
                abslx::Nanoseconds(1) / 2,
            sigfigs / 3);
  sigfigs = abslx::Seconds(int64_t{7000000000});
  EXPECT_EQ(abslx::Seconds(2333333333) + abslx::Nanoseconds(333333333) +
                abslx::Nanoseconds(1) / 4,
            sigfigs / 3);

  EXPECT_EQ(abslx::Seconds(7) + abslx::Milliseconds(500), abslx::Seconds(3) * 2.5);
  EXPECT_EQ(abslx::Seconds(8) * -1 + abslx::Milliseconds(300),
            (abslx::Seconds(2) + abslx::Milliseconds(200)) * -3.5);
  EXPECT_EQ(-abslx::Seconds(8) + abslx::Milliseconds(300),
            (abslx::Seconds(2) + abslx::Milliseconds(200)) * -3.5);
  EXPECT_EQ(abslx::Seconds(1) + abslx::Milliseconds(875),
            (abslx::Seconds(7) + abslx::Milliseconds(500)) / 4);
  EXPECT_EQ(abslx::Seconds(30),
            (abslx::Seconds(7) + abslx::Milliseconds(500)) / 0.25);
  EXPECT_EQ(abslx::Seconds(3),
            (abslx::Seconds(7) + abslx::Milliseconds(500)) / 2.5);

  // Tests division remainder.
  EXPECT_EQ(abslx::Nanoseconds(0), abslx::Nanoseconds(7) % abslx::Nanoseconds(1));
  EXPECT_EQ(abslx::Nanoseconds(0), abslx::Nanoseconds(0) % abslx::Nanoseconds(10));
  EXPECT_EQ(abslx::Nanoseconds(2), abslx::Nanoseconds(7) % abslx::Nanoseconds(5));
  EXPECT_EQ(abslx::Nanoseconds(2), abslx::Nanoseconds(2) % abslx::Nanoseconds(5));

  EXPECT_EQ(abslx::Nanoseconds(1), abslx::Nanoseconds(10) % abslx::Nanoseconds(3));
  EXPECT_EQ(abslx::Nanoseconds(1),
            abslx::Nanoseconds(10) % abslx::Nanoseconds(-3));
  EXPECT_EQ(abslx::Nanoseconds(-1),
            abslx::Nanoseconds(-10) % abslx::Nanoseconds(3));
  EXPECT_EQ(abslx::Nanoseconds(-1),
            abslx::Nanoseconds(-10) % abslx::Nanoseconds(-3));

  EXPECT_EQ(abslx::Milliseconds(100),
            abslx::Seconds(1) % abslx::Milliseconds(300));
  EXPECT_EQ(
      abslx::Milliseconds(300),
      (abslx::Seconds(3) + abslx::Milliseconds(800)) % abslx::Milliseconds(500));

  EXPECT_EQ(abslx::Nanoseconds(1), abslx::Nanoseconds(1) % abslx::Seconds(1));
  EXPECT_EQ(abslx::Nanoseconds(-1), abslx::Nanoseconds(-1) % abslx::Seconds(1));
  EXPECT_EQ(0, abslx::Nanoseconds(-1) / abslx::Seconds(1));  // Actual -1e-9

  // Tests identity a = (a/b)*b + a%b
#define TEST_MOD_IDENTITY(a, b) \
  EXPECT_EQ((a), ((a) / (b))*(b) + ((a)%(b)))

  TEST_MOD_IDENTITY(abslx::Seconds(0), abslx::Seconds(2));
  TEST_MOD_IDENTITY(abslx::Seconds(1), abslx::Seconds(1));
  TEST_MOD_IDENTITY(abslx::Seconds(1), abslx::Seconds(2));
  TEST_MOD_IDENTITY(abslx::Seconds(2), abslx::Seconds(1));

  TEST_MOD_IDENTITY(abslx::Seconds(-2), abslx::Seconds(1));
  TEST_MOD_IDENTITY(abslx::Seconds(2), abslx::Seconds(-1));
  TEST_MOD_IDENTITY(abslx::Seconds(-2), abslx::Seconds(-1));

  TEST_MOD_IDENTITY(abslx::Nanoseconds(0), abslx::Nanoseconds(2));
  TEST_MOD_IDENTITY(abslx::Nanoseconds(1), abslx::Nanoseconds(1));
  TEST_MOD_IDENTITY(abslx::Nanoseconds(1), abslx::Nanoseconds(2));
  TEST_MOD_IDENTITY(abslx::Nanoseconds(2), abslx::Nanoseconds(1));

  TEST_MOD_IDENTITY(abslx::Nanoseconds(-2), abslx::Nanoseconds(1));
  TEST_MOD_IDENTITY(abslx::Nanoseconds(2), abslx::Nanoseconds(-1));
  TEST_MOD_IDENTITY(abslx::Nanoseconds(-2), abslx::Nanoseconds(-1));

  // Mixed seconds + subseconds
  abslx::Duration mixed_a = abslx::Seconds(1) + abslx::Nanoseconds(2);
  abslx::Duration mixed_b = abslx::Seconds(1) + abslx::Nanoseconds(3);

  TEST_MOD_IDENTITY(abslx::Seconds(0), mixed_a);
  TEST_MOD_IDENTITY(mixed_a, mixed_a);
  TEST_MOD_IDENTITY(mixed_a, mixed_b);
  TEST_MOD_IDENTITY(mixed_b, mixed_a);

  TEST_MOD_IDENTITY(-mixed_a, mixed_b);
  TEST_MOD_IDENTITY(mixed_a, -mixed_b);
  TEST_MOD_IDENTITY(-mixed_a, -mixed_b);

#undef TEST_MOD_IDENTITY
}

TEST(Duration, Truncation) {
  const abslx::Duration d = abslx::Nanoseconds(1234567890);
  const abslx::Duration inf = abslx::InfiniteDuration();
  for (int unit_sign : {1, -1}) {  // sign shouldn't matter
    EXPECT_EQ(abslx::Nanoseconds(1234567890),
              Trunc(d, unit_sign * abslx::Nanoseconds(1)));
    EXPECT_EQ(abslx::Microseconds(1234567),
              Trunc(d, unit_sign * abslx::Microseconds(1)));
    EXPECT_EQ(abslx::Milliseconds(1234),
              Trunc(d, unit_sign * abslx::Milliseconds(1)));
    EXPECT_EQ(abslx::Seconds(1), Trunc(d, unit_sign * abslx::Seconds(1)));
    EXPECT_EQ(inf, Trunc(inf, unit_sign * abslx::Seconds(1)));

    EXPECT_EQ(abslx::Nanoseconds(-1234567890),
              Trunc(-d, unit_sign * abslx::Nanoseconds(1)));
    EXPECT_EQ(abslx::Microseconds(-1234567),
              Trunc(-d, unit_sign * abslx::Microseconds(1)));
    EXPECT_EQ(abslx::Milliseconds(-1234),
              Trunc(-d, unit_sign * abslx::Milliseconds(1)));
    EXPECT_EQ(abslx::Seconds(-1), Trunc(-d, unit_sign * abslx::Seconds(1)));
    EXPECT_EQ(-inf, Trunc(-inf, unit_sign * abslx::Seconds(1)));
  }
}

TEST(Duration, Flooring) {
  const abslx::Duration d = abslx::Nanoseconds(1234567890);
  const abslx::Duration inf = abslx::InfiniteDuration();
  for (int unit_sign : {1, -1}) {  // sign shouldn't matter
    EXPECT_EQ(abslx::Nanoseconds(1234567890),
              abslx::Floor(d, unit_sign * abslx::Nanoseconds(1)));
    EXPECT_EQ(abslx::Microseconds(1234567),
              abslx::Floor(d, unit_sign * abslx::Microseconds(1)));
    EXPECT_EQ(abslx::Milliseconds(1234),
              abslx::Floor(d, unit_sign * abslx::Milliseconds(1)));
    EXPECT_EQ(abslx::Seconds(1), abslx::Floor(d, unit_sign * abslx::Seconds(1)));
    EXPECT_EQ(inf, abslx::Floor(inf, unit_sign * abslx::Seconds(1)));

    EXPECT_EQ(abslx::Nanoseconds(-1234567890),
              abslx::Floor(-d, unit_sign * abslx::Nanoseconds(1)));
    EXPECT_EQ(abslx::Microseconds(-1234568),
              abslx::Floor(-d, unit_sign * abslx::Microseconds(1)));
    EXPECT_EQ(abslx::Milliseconds(-1235),
              abslx::Floor(-d, unit_sign * abslx::Milliseconds(1)));
    EXPECT_EQ(abslx::Seconds(-2), abslx::Floor(-d, unit_sign * abslx::Seconds(1)));
    EXPECT_EQ(-inf, abslx::Floor(-inf, unit_sign * abslx::Seconds(1)));
  }
}

TEST(Duration, Ceiling) {
  const abslx::Duration d = abslx::Nanoseconds(1234567890);
  const abslx::Duration inf = abslx::InfiniteDuration();
  for (int unit_sign : {1, -1}) {  // // sign shouldn't matter
    EXPECT_EQ(abslx::Nanoseconds(1234567890),
              abslx::Ceil(d, unit_sign * abslx::Nanoseconds(1)));
    EXPECT_EQ(abslx::Microseconds(1234568),
              abslx::Ceil(d, unit_sign * abslx::Microseconds(1)));
    EXPECT_EQ(abslx::Milliseconds(1235),
              abslx::Ceil(d, unit_sign * abslx::Milliseconds(1)));
    EXPECT_EQ(abslx::Seconds(2), abslx::Ceil(d, unit_sign * abslx::Seconds(1)));
    EXPECT_EQ(inf, abslx::Ceil(inf, unit_sign * abslx::Seconds(1)));

    EXPECT_EQ(abslx::Nanoseconds(-1234567890),
              abslx::Ceil(-d, unit_sign * abslx::Nanoseconds(1)));
    EXPECT_EQ(abslx::Microseconds(-1234567),
              abslx::Ceil(-d, unit_sign * abslx::Microseconds(1)));
    EXPECT_EQ(abslx::Milliseconds(-1234),
              abslx::Ceil(-d, unit_sign * abslx::Milliseconds(1)));
    EXPECT_EQ(abslx::Seconds(-1), abslx::Ceil(-d, unit_sign * abslx::Seconds(1)));
    EXPECT_EQ(-inf, abslx::Ceil(-inf, unit_sign * abslx::Seconds(1)));
  }
}

TEST(Duration, RoundTripUnits) {
  const int kRange = 100000;

#define ROUND_TRIP_UNIT(U, LOW, HIGH)          \
  do {                                         \
    for (int64_t i = LOW; i < HIGH; ++i) {     \
      abslx::Duration d = abslx::U(i);           \
      if (d == abslx::InfiniteDuration())       \
        EXPECT_EQ(kint64max, d / abslx::U(1));  \
      else if (d == -abslx::InfiniteDuration()) \
        EXPECT_EQ(kint64min, d / abslx::U(1));  \
      else                                     \
        EXPECT_EQ(i, abslx::U(i) / abslx::U(1)); \
    }                                          \
  } while (0)

  ROUND_TRIP_UNIT(Nanoseconds, kint64min, kint64min + kRange);
  ROUND_TRIP_UNIT(Nanoseconds, -kRange, kRange);
  ROUND_TRIP_UNIT(Nanoseconds, kint64max - kRange, kint64max);

  ROUND_TRIP_UNIT(Microseconds, kint64min, kint64min + kRange);
  ROUND_TRIP_UNIT(Microseconds, -kRange, kRange);
  ROUND_TRIP_UNIT(Microseconds, kint64max - kRange, kint64max);

  ROUND_TRIP_UNIT(Milliseconds, kint64min, kint64min + kRange);
  ROUND_TRIP_UNIT(Milliseconds, -kRange, kRange);
  ROUND_TRIP_UNIT(Milliseconds, kint64max - kRange, kint64max);

  ROUND_TRIP_UNIT(Seconds, kint64min, kint64min + kRange);
  ROUND_TRIP_UNIT(Seconds, -kRange, kRange);
  ROUND_TRIP_UNIT(Seconds, kint64max - kRange, kint64max);

  ROUND_TRIP_UNIT(Minutes, kint64min / 60, kint64min / 60 + kRange);
  ROUND_TRIP_UNIT(Minutes, -kRange, kRange);
  ROUND_TRIP_UNIT(Minutes, kint64max / 60 - kRange, kint64max / 60);

  ROUND_TRIP_UNIT(Hours, kint64min / 3600, kint64min / 3600 + kRange);
  ROUND_TRIP_UNIT(Hours, -kRange, kRange);
  ROUND_TRIP_UNIT(Hours, kint64max / 3600 - kRange, kint64max / 3600);

#undef ROUND_TRIP_UNIT
}

TEST(Duration, TruncConversions) {
  // Tests ToTimespec()/DurationFromTimespec()
  const struct {
    abslx::Duration d;
    timespec ts;
  } to_ts[] = {
      {abslx::Seconds(1) + abslx::Nanoseconds(1), {1, 1}},
      {abslx::Seconds(1) + abslx::Nanoseconds(1) / 2, {1, 0}},
      {abslx::Seconds(1) + abslx::Nanoseconds(0), {1, 0}},
      {abslx::Seconds(0) + abslx::Nanoseconds(0), {0, 0}},
      {abslx::Seconds(0) - abslx::Nanoseconds(1) / 2, {0, 0}},
      {abslx::Seconds(0) - abslx::Nanoseconds(1), {-1, 999999999}},
      {abslx::Seconds(-1) + abslx::Nanoseconds(1), {-1, 1}},
      {abslx::Seconds(-1) + abslx::Nanoseconds(1) / 2, {-1, 1}},
      {abslx::Seconds(-1) + abslx::Nanoseconds(0), {-1, 0}},
      {abslx::Seconds(-1) - abslx::Nanoseconds(1) / 2, {-1, 0}},
  };
  for (const auto& test : to_ts) {
    EXPECT_THAT(abslx::ToTimespec(test.d), TimespecMatcher(test.ts));
  }
  const struct {
    timespec ts;
    abslx::Duration d;
  } from_ts[] = {
      {{1, 1}, abslx::Seconds(1) + abslx::Nanoseconds(1)},
      {{1, 0}, abslx::Seconds(1) + abslx::Nanoseconds(0)},
      {{0, 0}, abslx::Seconds(0) + abslx::Nanoseconds(0)},
      {{0, -1}, abslx::Seconds(0) - abslx::Nanoseconds(1)},
      {{-1, 999999999}, abslx::Seconds(0) - abslx::Nanoseconds(1)},
      {{-1, 1}, abslx::Seconds(-1) + abslx::Nanoseconds(1)},
      {{-1, 0}, abslx::Seconds(-1) + abslx::Nanoseconds(0)},
      {{-1, -1}, abslx::Seconds(-1) - abslx::Nanoseconds(1)},
      {{-2, 999999999}, abslx::Seconds(-1) - abslx::Nanoseconds(1)},
  };
  for (const auto& test : from_ts) {
    EXPECT_EQ(test.d, abslx::DurationFromTimespec(test.ts));
  }

  // Tests ToTimeval()/DurationFromTimeval() (same as timespec above)
  const struct {
    abslx::Duration d;
    timeval tv;
  } to_tv[] = {
      {abslx::Seconds(1) + abslx::Microseconds(1), {1, 1}},
      {abslx::Seconds(1) + abslx::Microseconds(1) / 2, {1, 0}},
      {abslx::Seconds(1) + abslx::Microseconds(0), {1, 0}},
      {abslx::Seconds(0) + abslx::Microseconds(0), {0, 0}},
      {abslx::Seconds(0) - abslx::Microseconds(1) / 2, {0, 0}},
      {abslx::Seconds(0) - abslx::Microseconds(1), {-1, 999999}},
      {abslx::Seconds(-1) + abslx::Microseconds(1), {-1, 1}},
      {abslx::Seconds(-1) + abslx::Microseconds(1) / 2, {-1, 1}},
      {abslx::Seconds(-1) + abslx::Microseconds(0), {-1, 0}},
      {abslx::Seconds(-1) - abslx::Microseconds(1) / 2, {-1, 0}},
  };
  for (const auto& test : to_tv) {
    EXPECT_THAT(abslx::ToTimeval(test.d), TimevalMatcher(test.tv));
  }
  const struct {
    timeval tv;
    abslx::Duration d;
  } from_tv[] = {
      {{1, 1}, abslx::Seconds(1) + abslx::Microseconds(1)},
      {{1, 0}, abslx::Seconds(1) + abslx::Microseconds(0)},
      {{0, 0}, abslx::Seconds(0) + abslx::Microseconds(0)},
      {{0, -1}, abslx::Seconds(0) - abslx::Microseconds(1)},
      {{-1, 999999}, abslx::Seconds(0) - abslx::Microseconds(1)},
      {{-1, 1}, abslx::Seconds(-1) + abslx::Microseconds(1)},
      {{-1, 0}, abslx::Seconds(-1) + abslx::Microseconds(0)},
      {{-1, -1}, abslx::Seconds(-1) - abslx::Microseconds(1)},
      {{-2, 999999}, abslx::Seconds(-1) - abslx::Microseconds(1)},
  };
  for (const auto& test : from_tv) {
    EXPECT_EQ(test.d, abslx::DurationFromTimeval(test.tv));
  }
}

TEST(Duration, SmallConversions) {
  // Special tests for conversions of small durations.

  EXPECT_EQ(abslx::ZeroDuration(), abslx::Seconds(0));
  // TODO(bww): Is the next one OK?
  EXPECT_EQ(abslx::ZeroDuration(), abslx::Seconds(0.124999999e-9));
  EXPECT_EQ(abslx::Nanoseconds(1) / 4, abslx::Seconds(0.125e-9));
  EXPECT_EQ(abslx::Nanoseconds(1) / 4, abslx::Seconds(0.250e-9));
  EXPECT_EQ(abslx::Nanoseconds(1) / 2, abslx::Seconds(0.375e-9));
  EXPECT_EQ(abslx::Nanoseconds(1) / 2, abslx::Seconds(0.500e-9));
  EXPECT_EQ(abslx::Nanoseconds(3) / 4, abslx::Seconds(0.625e-9));
  EXPECT_EQ(abslx::Nanoseconds(3) / 4, abslx::Seconds(0.750e-9));
  EXPECT_EQ(abslx::Nanoseconds(1), abslx::Seconds(0.875e-9));
  EXPECT_EQ(abslx::Nanoseconds(1), abslx::Seconds(1.000e-9));

  EXPECT_EQ(abslx::ZeroDuration(), abslx::Seconds(-0.124999999e-9));
  EXPECT_EQ(-abslx::Nanoseconds(1) / 4, abslx::Seconds(-0.125e-9));
  EXPECT_EQ(-abslx::Nanoseconds(1) / 4, abslx::Seconds(-0.250e-9));
  EXPECT_EQ(-abslx::Nanoseconds(1) / 2, abslx::Seconds(-0.375e-9));
  EXPECT_EQ(-abslx::Nanoseconds(1) / 2, abslx::Seconds(-0.500e-9));
  EXPECT_EQ(-abslx::Nanoseconds(3) / 4, abslx::Seconds(-0.625e-9));
  EXPECT_EQ(-abslx::Nanoseconds(3) / 4, abslx::Seconds(-0.750e-9));
  EXPECT_EQ(-abslx::Nanoseconds(1), abslx::Seconds(-0.875e-9));
  EXPECT_EQ(-abslx::Nanoseconds(1), abslx::Seconds(-1.000e-9));

  timespec ts;
  ts.tv_sec = 0;
  ts.tv_nsec = 0;
  EXPECT_THAT(ToTimespec(abslx::Nanoseconds(0)), TimespecMatcher(ts));
  // TODO(bww): Are the next three OK?
  EXPECT_THAT(ToTimespec(abslx::Nanoseconds(1) / 4), TimespecMatcher(ts));
  EXPECT_THAT(ToTimespec(abslx::Nanoseconds(2) / 4), TimespecMatcher(ts));
  EXPECT_THAT(ToTimespec(abslx::Nanoseconds(3) / 4), TimespecMatcher(ts));
  ts.tv_nsec = 1;
  EXPECT_THAT(ToTimespec(abslx::Nanoseconds(4) / 4), TimespecMatcher(ts));
  EXPECT_THAT(ToTimespec(abslx::Nanoseconds(5) / 4), TimespecMatcher(ts));
  EXPECT_THAT(ToTimespec(abslx::Nanoseconds(6) / 4), TimespecMatcher(ts));
  EXPECT_THAT(ToTimespec(abslx::Nanoseconds(7) / 4), TimespecMatcher(ts));
  ts.tv_nsec = 2;
  EXPECT_THAT(ToTimespec(abslx::Nanoseconds(8) / 4), TimespecMatcher(ts));

  timeval tv;
  tv.tv_sec = 0;
  tv.tv_usec = 0;
  EXPECT_THAT(ToTimeval(abslx::Nanoseconds(0)), TimevalMatcher(tv));
  // TODO(bww): Is the next one OK?
  EXPECT_THAT(ToTimeval(abslx::Nanoseconds(999)), TimevalMatcher(tv));
  tv.tv_usec = 1;
  EXPECT_THAT(ToTimeval(abslx::Nanoseconds(1000)), TimevalMatcher(tv));
  EXPECT_THAT(ToTimeval(abslx::Nanoseconds(1999)), TimevalMatcher(tv));
  tv.tv_usec = 2;
  EXPECT_THAT(ToTimeval(abslx::Nanoseconds(2000)), TimevalMatcher(tv));
}

void VerifyApproxSameAsMul(double time_as_seconds, int* const misses) {
  auto direct_seconds = abslx::Seconds(time_as_seconds);
  auto mul_by_one_second = time_as_seconds * abslx::Seconds(1);
  // These are expected to differ by up to one tick due to fused multiply/add
  // contraction.
  if (abslx::AbsDuration(direct_seconds - mul_by_one_second) >
      abslx::time_internal::MakeDuration(0, 1u)) {
    if (*misses > 10) return;
    ASSERT_LE(++(*misses), 10) << "Too many errors, not reporting more.";
    EXPECT_EQ(direct_seconds, mul_by_one_second)
        << "given double time_as_seconds = " << std::setprecision(17)
        << time_as_seconds;
  }
}

// For a variety of interesting durations, we find the exact point
// where one double converts to that duration, and the very next double
// converts to the next duration.  For both of those points, verify that
// Seconds(point) returns a duration near point * Seconds(1.0). (They may
// not be exactly equal due to fused multiply/add contraction.)
TEST(Duration, ToDoubleSecondsCheckEdgeCases) {
  constexpr uint32_t kTicksPerSecond = abslx::time_internal::kTicksPerSecond;
  constexpr auto duration_tick = abslx::time_internal::MakeDuration(0, 1u);
  int misses = 0;
  for (int64_t seconds = 0; seconds < 99; ++seconds) {
    uint32_t tick_vals[] = {0, +999, +999999, +999999999, kTicksPerSecond - 1,
                            0, 1000, 1000000, 1000000000, kTicksPerSecond,
                            1, 1001, 1000001, 1000000001, kTicksPerSecond + 1,
                            2, 1002, 1000002, 1000000002, kTicksPerSecond + 2,
                            3, 1003, 1000003, 1000000003, kTicksPerSecond + 3,
                            4, 1004, 1000004, 1000000004, kTicksPerSecond + 4,
                            5, 6,    7,       8,          9};
    for (uint32_t ticks : tick_vals) {
      abslx::Duration s_plus_t = abslx::Seconds(seconds) + ticks * duration_tick;
      for (abslx::Duration d : {s_plus_t, -s_plus_t}) {
        abslx::Duration after_d = d + duration_tick;
        EXPECT_NE(d, after_d);
        EXPECT_EQ(after_d - d, duration_tick);

        double low_edge = ToDoubleSeconds(d);
        EXPECT_EQ(d, abslx::Seconds(low_edge));

        double high_edge = ToDoubleSeconds(after_d);
        EXPECT_EQ(after_d, abslx::Seconds(high_edge));

        for (;;) {
          double midpoint = low_edge + (high_edge - low_edge) / 2;
          if (midpoint == low_edge || midpoint == high_edge) break;
          abslx::Duration mid_duration = abslx::Seconds(midpoint);
          if (mid_duration == d) {
            low_edge = midpoint;
          } else {
            EXPECT_EQ(mid_duration, after_d);
            high_edge = midpoint;
          }
        }
        // Now low_edge is the highest double that converts to Duration d,
        // and high_edge is the lowest double that converts to Duration after_d.
        VerifyApproxSameAsMul(low_edge, &misses);
        VerifyApproxSameAsMul(high_edge, &misses);
      }
    }
  }
}

TEST(Duration, ToDoubleSecondsCheckRandom) {
  std::random_device rd;
  std::seed_seq seed({rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()});
  std::mt19937_64 gen(seed);
  // We want doubles distributed from 1/8ns up to 2^63, where
  // as many values are tested from 1ns to 2ns as from 1sec to 2sec,
  // so even distribute along a log-scale of those values, and
  // exponentiate before using them.  (9.223377e+18 is just slightly
  // out of bounds for abslx::Duration.)
  std::uniform_real_distribution<double> uniform(std::log(0.125e-9),
                                                 std::log(9.223377e+18));
  int misses = 0;
  for (int i = 0; i < 1000000; ++i) {
    double d = std::exp(uniform(gen));
    VerifyApproxSameAsMul(d, &misses);
    VerifyApproxSameAsMul(-d, &misses);
  }
}

TEST(Duration, ConversionSaturation) {
  abslx::Duration d;

  const auto max_timeval_sec =
      std::numeric_limits<decltype(timeval::tv_sec)>::max();
  const auto min_timeval_sec =
      std::numeric_limits<decltype(timeval::tv_sec)>::min();
  timeval tv;
  tv.tv_sec = max_timeval_sec;
  tv.tv_usec = 999998;
  d = abslx::DurationFromTimeval(tv);
  tv = ToTimeval(d);
  EXPECT_EQ(max_timeval_sec, tv.tv_sec);
  EXPECT_EQ(999998, tv.tv_usec);
  d += abslx::Microseconds(1);
  tv = ToTimeval(d);
  EXPECT_EQ(max_timeval_sec, tv.tv_sec);
  EXPECT_EQ(999999, tv.tv_usec);
  d += abslx::Microseconds(1);  // no effect
  tv = ToTimeval(d);
  EXPECT_EQ(max_timeval_sec, tv.tv_sec);
  EXPECT_EQ(999999, tv.tv_usec);

  tv.tv_sec = min_timeval_sec;
  tv.tv_usec = 1;
  d = abslx::DurationFromTimeval(tv);
  tv = ToTimeval(d);
  EXPECT_EQ(min_timeval_sec, tv.tv_sec);
  EXPECT_EQ(1, tv.tv_usec);
  d -= abslx::Microseconds(1);
  tv = ToTimeval(d);
  EXPECT_EQ(min_timeval_sec, tv.tv_sec);
  EXPECT_EQ(0, tv.tv_usec);
  d -= abslx::Microseconds(1);  // no effect
  tv = ToTimeval(d);
  EXPECT_EQ(min_timeval_sec, tv.tv_sec);
  EXPECT_EQ(0, tv.tv_usec);

  const auto max_timespec_sec =
      std::numeric_limits<decltype(timespec::tv_sec)>::max();
  const auto min_timespec_sec =
      std::numeric_limits<decltype(timespec::tv_sec)>::min();
  timespec ts;
  ts.tv_sec = max_timespec_sec;
  ts.tv_nsec = 999999998;
  d = abslx::DurationFromTimespec(ts);
  ts = abslx::ToTimespec(d);
  EXPECT_EQ(max_timespec_sec, ts.tv_sec);
  EXPECT_EQ(999999998, ts.tv_nsec);
  d += abslx::Nanoseconds(1);
  ts = abslx::ToTimespec(d);
  EXPECT_EQ(max_timespec_sec, ts.tv_sec);
  EXPECT_EQ(999999999, ts.tv_nsec);
  d += abslx::Nanoseconds(1);  // no effect
  ts = abslx::ToTimespec(d);
  EXPECT_EQ(max_timespec_sec, ts.tv_sec);
  EXPECT_EQ(999999999, ts.tv_nsec);

  ts.tv_sec = min_timespec_sec;
  ts.tv_nsec = 1;
  d = abslx::DurationFromTimespec(ts);
  ts = abslx::ToTimespec(d);
  EXPECT_EQ(min_timespec_sec, ts.tv_sec);
  EXPECT_EQ(1, ts.tv_nsec);
  d -= abslx::Nanoseconds(1);
  ts = abslx::ToTimespec(d);
  EXPECT_EQ(min_timespec_sec, ts.tv_sec);
  EXPECT_EQ(0, ts.tv_nsec);
  d -= abslx::Nanoseconds(1);  // no effect
  ts = abslx::ToTimespec(d);
  EXPECT_EQ(min_timespec_sec, ts.tv_sec);
  EXPECT_EQ(0, ts.tv_nsec);
}

TEST(Duration, FormatDuration) {
  // Example from Go's docs.
  EXPECT_EQ("72h3m0.5s",
            abslx::FormatDuration(abslx::Hours(72) + abslx::Minutes(3) +
                                 abslx::Milliseconds(500)));
  // Go's largest time: 2540400h10m10.000000000s
  EXPECT_EQ("2540400h10m10s",
            abslx::FormatDuration(abslx::Hours(2540400) + abslx::Minutes(10) +
                                 abslx::Seconds(10)));

  EXPECT_EQ("0", abslx::FormatDuration(abslx::ZeroDuration()));
  EXPECT_EQ("0", abslx::FormatDuration(abslx::Seconds(0)));
  EXPECT_EQ("0", abslx::FormatDuration(abslx::Nanoseconds(0)));

  EXPECT_EQ("1ns", abslx::FormatDuration(abslx::Nanoseconds(1)));
  EXPECT_EQ("1us", abslx::FormatDuration(abslx::Microseconds(1)));
  EXPECT_EQ("1ms", abslx::FormatDuration(abslx::Milliseconds(1)));
  EXPECT_EQ("1s", abslx::FormatDuration(abslx::Seconds(1)));
  EXPECT_EQ("1m", abslx::FormatDuration(abslx::Minutes(1)));
  EXPECT_EQ("1h", abslx::FormatDuration(abslx::Hours(1)));

  EXPECT_EQ("1h1m", abslx::FormatDuration(abslx::Hours(1) + abslx::Minutes(1)));
  EXPECT_EQ("1h1s", abslx::FormatDuration(abslx::Hours(1) + abslx::Seconds(1)));
  EXPECT_EQ("1m1s", abslx::FormatDuration(abslx::Minutes(1) + abslx::Seconds(1)));

  EXPECT_EQ("1h0.25s",
            abslx::FormatDuration(abslx::Hours(1) + abslx::Milliseconds(250)));
  EXPECT_EQ("1m0.25s",
            abslx::FormatDuration(abslx::Minutes(1) + abslx::Milliseconds(250)));
  EXPECT_EQ("1h1m0.25s",
            abslx::FormatDuration(abslx::Hours(1) + abslx::Minutes(1) +
                                 abslx::Milliseconds(250)));
  EXPECT_EQ("1h0.0005s",
            abslx::FormatDuration(abslx::Hours(1) + abslx::Microseconds(500)));
  EXPECT_EQ("1h0.0000005s",
            abslx::FormatDuration(abslx::Hours(1) + abslx::Nanoseconds(500)));

  // Subsecond special case.
  EXPECT_EQ("1.5ns", abslx::FormatDuration(abslx::Nanoseconds(1) +
                                          abslx::Nanoseconds(1) / 2));
  EXPECT_EQ("1.25ns", abslx::FormatDuration(abslx::Nanoseconds(1) +
                                           abslx::Nanoseconds(1) / 4));
  EXPECT_EQ("1ns", abslx::FormatDuration(abslx::Nanoseconds(1) +
                                        abslx::Nanoseconds(1) / 9));
  EXPECT_EQ("1.2us", abslx::FormatDuration(abslx::Microseconds(1) +
                                          abslx::Nanoseconds(200)));
  EXPECT_EQ("1.2ms", abslx::FormatDuration(abslx::Milliseconds(1) +
                                          abslx::Microseconds(200)));
  EXPECT_EQ("1.0002ms", abslx::FormatDuration(abslx::Milliseconds(1) +
                                             abslx::Nanoseconds(200)));
  EXPECT_EQ("1.00001ms", abslx::FormatDuration(abslx::Milliseconds(1) +
                                              abslx::Nanoseconds(10)));
  EXPECT_EQ("1.000001ms",
            abslx::FormatDuration(abslx::Milliseconds(1) + abslx::Nanoseconds(1)));

  // Negative durations.
  EXPECT_EQ("-1ns", abslx::FormatDuration(abslx::Nanoseconds(-1)));
  EXPECT_EQ("-1us", abslx::FormatDuration(abslx::Microseconds(-1)));
  EXPECT_EQ("-1ms", abslx::FormatDuration(abslx::Milliseconds(-1)));
  EXPECT_EQ("-1s", abslx::FormatDuration(abslx::Seconds(-1)));
  EXPECT_EQ("-1m", abslx::FormatDuration(abslx::Minutes(-1)));
  EXPECT_EQ("-1h", abslx::FormatDuration(abslx::Hours(-1)));

  EXPECT_EQ("-1h1m",
            abslx::FormatDuration(-(abslx::Hours(1) + abslx::Minutes(1))));
  EXPECT_EQ("-1h1s",
            abslx::FormatDuration(-(abslx::Hours(1) + abslx::Seconds(1))));
  EXPECT_EQ("-1m1s",
            abslx::FormatDuration(-(abslx::Minutes(1) + abslx::Seconds(1))));

  EXPECT_EQ("-1ns", abslx::FormatDuration(abslx::Nanoseconds(-1)));
  EXPECT_EQ("-1.2us", abslx::FormatDuration(
                          -(abslx::Microseconds(1) + abslx::Nanoseconds(200))));
  EXPECT_EQ("-1.2ms", abslx::FormatDuration(
                          -(abslx::Milliseconds(1) + abslx::Microseconds(200))));
  EXPECT_EQ("-1.0002ms", abslx::FormatDuration(-(abslx::Milliseconds(1) +
                                                abslx::Nanoseconds(200))));
  EXPECT_EQ("-1.00001ms", abslx::FormatDuration(-(abslx::Milliseconds(1) +
                                                 abslx::Nanoseconds(10))));
  EXPECT_EQ("-1.000001ms", abslx::FormatDuration(-(abslx::Milliseconds(1) +
                                                  abslx::Nanoseconds(1))));

  //
  // Interesting corner cases.
  //

  const abslx::Duration qns = abslx::Nanoseconds(1) / 4;
  const abslx::Duration max_dur =
      abslx::Seconds(kint64max) + (abslx::Seconds(1) - qns);
  const abslx::Duration min_dur = abslx::Seconds(kint64min);

  EXPECT_EQ("0.25ns", abslx::FormatDuration(qns));
  EXPECT_EQ("-0.25ns", abslx::FormatDuration(-qns));
  EXPECT_EQ("2562047788015215h30m7.99999999975s",
            abslx::FormatDuration(max_dur));
  EXPECT_EQ("-2562047788015215h30m8s", abslx::FormatDuration(min_dur));

  // Tests printing full precision from units that print using FDivDuration
  EXPECT_EQ("55.00000000025s", abslx::FormatDuration(abslx::Seconds(55) + qns));
  EXPECT_EQ("55.00000025ms",
            abslx::FormatDuration(abslx::Milliseconds(55) + qns));
  EXPECT_EQ("55.00025us", abslx::FormatDuration(abslx::Microseconds(55) + qns));
  EXPECT_EQ("55.25ns", abslx::FormatDuration(abslx::Nanoseconds(55) + qns));

  // Formatting infinity
  EXPECT_EQ("inf", abslx::FormatDuration(abslx::InfiniteDuration()));
  EXPECT_EQ("-inf", abslx::FormatDuration(-abslx::InfiniteDuration()));

  // Formatting approximately +/- 100 billion years
  const abslx::Duration huge_range = ApproxYears(100000000000);
  EXPECT_EQ("876000000000000h", abslx::FormatDuration(huge_range));
  EXPECT_EQ("-876000000000000h", abslx::FormatDuration(-huge_range));

  EXPECT_EQ("876000000000000h0.999999999s",
            abslx::FormatDuration(huge_range +
                                 (abslx::Seconds(1) - abslx::Nanoseconds(1))));
  EXPECT_EQ("876000000000000h0.9999999995s",
            abslx::FormatDuration(
                huge_range + (abslx::Seconds(1) - abslx::Nanoseconds(1) / 2)));
  EXPECT_EQ("876000000000000h0.99999999975s",
            abslx::FormatDuration(
                huge_range + (abslx::Seconds(1) - abslx::Nanoseconds(1) / 4)));

  EXPECT_EQ("-876000000000000h0.999999999s",
            abslx::FormatDuration(-huge_range -
                                 (abslx::Seconds(1) - abslx::Nanoseconds(1))));
  EXPECT_EQ("-876000000000000h0.9999999995s",
            abslx::FormatDuration(
                -huge_range - (abslx::Seconds(1) - abslx::Nanoseconds(1) / 2)));
  EXPECT_EQ("-876000000000000h0.99999999975s",
            abslx::FormatDuration(
                -huge_range - (abslx::Seconds(1) - abslx::Nanoseconds(1) / 4)));
}

TEST(Duration, ParseDuration) {
  abslx::Duration d;

  // No specified unit. Should only work for zero and infinity.
  EXPECT_TRUE(abslx::ParseDuration("0", &d));
  EXPECT_EQ(abslx::ZeroDuration(), d);
  EXPECT_TRUE(abslx::ParseDuration("+0", &d));
  EXPECT_EQ(abslx::ZeroDuration(), d);
  EXPECT_TRUE(abslx::ParseDuration("-0", &d));
  EXPECT_EQ(abslx::ZeroDuration(), d);

  EXPECT_TRUE(abslx::ParseDuration("inf", &d));
  EXPECT_EQ(abslx::InfiniteDuration(), d);
  EXPECT_TRUE(abslx::ParseDuration("+inf", &d));
  EXPECT_EQ(abslx::InfiniteDuration(), d);
  EXPECT_TRUE(abslx::ParseDuration("-inf", &d));
  EXPECT_EQ(-abslx::InfiniteDuration(), d);
  EXPECT_FALSE(abslx::ParseDuration("infBlah", &d));

  // Illegal input forms.
  EXPECT_FALSE(abslx::ParseDuration("", &d));
  EXPECT_FALSE(abslx::ParseDuration("0.0", &d));
  EXPECT_FALSE(abslx::ParseDuration(".0", &d));
  EXPECT_FALSE(abslx::ParseDuration(".", &d));
  EXPECT_FALSE(abslx::ParseDuration("01", &d));
  EXPECT_FALSE(abslx::ParseDuration("1", &d));
  EXPECT_FALSE(abslx::ParseDuration("-1", &d));
  EXPECT_FALSE(abslx::ParseDuration("2", &d));
  EXPECT_FALSE(abslx::ParseDuration("2 s", &d));
  EXPECT_FALSE(abslx::ParseDuration(".s", &d));
  EXPECT_FALSE(abslx::ParseDuration("-.s", &d));
  EXPECT_FALSE(abslx::ParseDuration("s", &d));
  EXPECT_FALSE(abslx::ParseDuration(" 2s", &d));
  EXPECT_FALSE(abslx::ParseDuration("2s ", &d));
  EXPECT_FALSE(abslx::ParseDuration(" 2s ", &d));
  EXPECT_FALSE(abslx::ParseDuration("2mt", &d));
  EXPECT_FALSE(abslx::ParseDuration("1e3s", &d));

  // One unit type.
  EXPECT_TRUE(abslx::ParseDuration("1ns", &d));
  EXPECT_EQ(abslx::Nanoseconds(1), d);
  EXPECT_TRUE(abslx::ParseDuration("1us", &d));
  EXPECT_EQ(abslx::Microseconds(1), d);
  EXPECT_TRUE(abslx::ParseDuration("1ms", &d));
  EXPECT_EQ(abslx::Milliseconds(1), d);
  EXPECT_TRUE(abslx::ParseDuration("1s", &d));
  EXPECT_EQ(abslx::Seconds(1), d);
  EXPECT_TRUE(abslx::ParseDuration("2m", &d));
  EXPECT_EQ(abslx::Minutes(2), d);
  EXPECT_TRUE(abslx::ParseDuration("2h", &d));
  EXPECT_EQ(abslx::Hours(2), d);

  // Huge counts of a unit.
  EXPECT_TRUE(abslx::ParseDuration("9223372036854775807us", &d));
  EXPECT_EQ(abslx::Microseconds(9223372036854775807), d);
  EXPECT_TRUE(abslx::ParseDuration("-9223372036854775807us", &d));
  EXPECT_EQ(abslx::Microseconds(-9223372036854775807), d);

  // Multiple units.
  EXPECT_TRUE(abslx::ParseDuration("2h3m4s", &d));
  EXPECT_EQ(abslx::Hours(2) + abslx::Minutes(3) + abslx::Seconds(4), d);
  EXPECT_TRUE(abslx::ParseDuration("3m4s5us", &d));
  EXPECT_EQ(abslx::Minutes(3) + abslx::Seconds(4) + abslx::Microseconds(5), d);
  EXPECT_TRUE(abslx::ParseDuration("2h3m4s5ms6us7ns", &d));
  EXPECT_EQ(abslx::Hours(2) + abslx::Minutes(3) + abslx::Seconds(4) +
                abslx::Milliseconds(5) + abslx::Microseconds(6) +
                abslx::Nanoseconds(7),
            d);

  // Multiple units out of order.
  EXPECT_TRUE(abslx::ParseDuration("2us3m4s5h", &d));
  EXPECT_EQ(abslx::Hours(5) + abslx::Minutes(3) + abslx::Seconds(4) +
                abslx::Microseconds(2),
            d);

  // Fractional values of units.
  EXPECT_TRUE(abslx::ParseDuration("1.5ns", &d));
  EXPECT_EQ(1.5 * abslx::Nanoseconds(1), d);
  EXPECT_TRUE(abslx::ParseDuration("1.5us", &d));
  EXPECT_EQ(1.5 * abslx::Microseconds(1), d);
  EXPECT_TRUE(abslx::ParseDuration("1.5ms", &d));
  EXPECT_EQ(1.5 * abslx::Milliseconds(1), d);
  EXPECT_TRUE(abslx::ParseDuration("1.5s", &d));
  EXPECT_EQ(1.5 * abslx::Seconds(1), d);
  EXPECT_TRUE(abslx::ParseDuration("1.5m", &d));
  EXPECT_EQ(1.5 * abslx::Minutes(1), d);
  EXPECT_TRUE(abslx::ParseDuration("1.5h", &d));
  EXPECT_EQ(1.5 * abslx::Hours(1), d);

  // Huge fractional counts of a unit.
  EXPECT_TRUE(abslx::ParseDuration("0.4294967295s", &d));
  EXPECT_EQ(abslx::Nanoseconds(429496729) + abslx::Nanoseconds(1) / 2, d);
  EXPECT_TRUE(abslx::ParseDuration("0.429496729501234567890123456789s", &d));
  EXPECT_EQ(abslx::Nanoseconds(429496729) + abslx::Nanoseconds(1) / 2, d);

  // Negative durations.
  EXPECT_TRUE(abslx::ParseDuration("-1s", &d));
  EXPECT_EQ(abslx::Seconds(-1), d);
  EXPECT_TRUE(abslx::ParseDuration("-1m", &d));
  EXPECT_EQ(abslx::Minutes(-1), d);
  EXPECT_TRUE(abslx::ParseDuration("-1h", &d));
  EXPECT_EQ(abslx::Hours(-1), d);

  EXPECT_TRUE(abslx::ParseDuration("-1h2s", &d));
  EXPECT_EQ(-(abslx::Hours(1) + abslx::Seconds(2)), d);
  EXPECT_FALSE(abslx::ParseDuration("1h-2s", &d));
  EXPECT_FALSE(abslx::ParseDuration("-1h-2s", &d));
  EXPECT_FALSE(abslx::ParseDuration("-1h -2s", &d));
}

TEST(Duration, FormatParseRoundTrip) {
#define TEST_PARSE_ROUNDTRIP(d)                \
  do {                                         \
    std::string s = abslx::FormatDuration(d);   \
    abslx::Duration dur;                        \
    EXPECT_TRUE(abslx::ParseDuration(s, &dur)); \
    EXPECT_EQ(d, dur);                         \
  } while (0)

  TEST_PARSE_ROUNDTRIP(abslx::Nanoseconds(1));
  TEST_PARSE_ROUNDTRIP(abslx::Microseconds(1));
  TEST_PARSE_ROUNDTRIP(abslx::Milliseconds(1));
  TEST_PARSE_ROUNDTRIP(abslx::Seconds(1));
  TEST_PARSE_ROUNDTRIP(abslx::Minutes(1));
  TEST_PARSE_ROUNDTRIP(abslx::Hours(1));
  TEST_PARSE_ROUNDTRIP(abslx::Hours(1) + abslx::Nanoseconds(2));

  TEST_PARSE_ROUNDTRIP(abslx::Nanoseconds(-1));
  TEST_PARSE_ROUNDTRIP(abslx::Microseconds(-1));
  TEST_PARSE_ROUNDTRIP(abslx::Milliseconds(-1));
  TEST_PARSE_ROUNDTRIP(abslx::Seconds(-1));
  TEST_PARSE_ROUNDTRIP(abslx::Minutes(-1));
  TEST_PARSE_ROUNDTRIP(abslx::Hours(-1));

  TEST_PARSE_ROUNDTRIP(abslx::Hours(-1) + abslx::Nanoseconds(2));
  TEST_PARSE_ROUNDTRIP(abslx::Hours(1) + abslx::Nanoseconds(-2));
  TEST_PARSE_ROUNDTRIP(abslx::Hours(-1) + abslx::Nanoseconds(-2));

  TEST_PARSE_ROUNDTRIP(abslx::Nanoseconds(1) +
                       abslx::Nanoseconds(1) / 4);  // 1.25ns

  const abslx::Duration huge_range = ApproxYears(100000000000);
  TEST_PARSE_ROUNDTRIP(huge_range);
  TEST_PARSE_ROUNDTRIP(huge_range + (abslx::Seconds(1) - abslx::Nanoseconds(1)));

#undef TEST_PARSE_ROUNDTRIP
}

}  // namespace
