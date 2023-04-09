// Copyright 2018 The Abseil Authors.
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

#include "absl/time/time.h"

#if !defined(_WIN32)
#include <sys/time.h>
#endif  // _WIN32
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <ctime>
#include <memory>
#include <string>

#include "absl/time/clock.h"
#include "absl/time/internal/test_util.h"
#include "benchmark/benchmark.h"

namespace {

//
// Addition/Subtraction of a duration
//

void BM_Time_Arithmetic(benchmark::State& state) {
  const abslx::Duration nano = abslx::Nanoseconds(1);
  const abslx::Duration sec = abslx::Seconds(1);
  abslx::Time t = abslx::UnixEpoch();
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(t += nano);
    benchmark::DoNotOptimize(t -= sec);
  }
}
BENCHMARK(BM_Time_Arithmetic);

//
// Time difference
//

void BM_Time_Difference(benchmark::State& state) {
  abslx::Time start = abslx::Now();
  abslx::Time end = start + abslx::Nanoseconds(1);
  abslx::Duration diff;
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(diff += end - start);
  }
}
BENCHMARK(BM_Time_Difference);

//
// ToDateTime
//
// In each "ToDateTime" benchmark we switch between two instants
// separated by at least one transition in order to defeat any
// internal caching of previous results (e.g., see local_time_hint_).
//
// The "UTC" variants use UTC instead of the Google/local time zone.
//

void BM_Time_ToDateTime_Absl(benchmark::State& state) {
  const abslx::TimeZone tz =
      abslx::time_internal::LoadTimeZone("America/Los_Angeles");
  abslx::Time t = abslx::FromUnixSeconds(1384569027);
  abslx::Time t2 = abslx::FromUnixSeconds(1418962578);
  while (state.KeepRunning()) {
    std::swap(t, t2);
    t += abslx::Seconds(1);
    benchmark::DoNotOptimize(t.In(tz));
  }
}
BENCHMARK(BM_Time_ToDateTime_Absl);

void BM_Time_ToDateTime_Libc(benchmark::State& state) {
  // No timezone support, so just use localtime.
  time_t t = 1384569027;
  time_t t2 = 1418962578;
  while (state.KeepRunning()) {
    std::swap(t, t2);
    t += 1;
    struct tm tm;
#if !defined(_WIN32)
    benchmark::DoNotOptimize(localtime_r(&t, &tm));
#else   // _WIN32
    benchmark::DoNotOptimize(localtime_s(&tm, &t));
#endif  // _WIN32
  }
}
BENCHMARK(BM_Time_ToDateTime_Libc);

void BM_Time_ToDateTimeUTC_Absl(benchmark::State& state) {
  const abslx::TimeZone tz = abslx::UTCTimeZone();
  abslx::Time t = abslx::FromUnixSeconds(1384569027);
  while (state.KeepRunning()) {
    t += abslx::Seconds(1);
    benchmark::DoNotOptimize(t.In(tz));
  }
}
BENCHMARK(BM_Time_ToDateTimeUTC_Absl);

void BM_Time_ToDateTimeUTC_Libc(benchmark::State& state) {
  time_t t = 1384569027;
  while (state.KeepRunning()) {
    t += 1;
    struct tm tm;
#if !defined(_WIN32)
    benchmark::DoNotOptimize(gmtime_r(&t, &tm));
#else   // _WIN32
    benchmark::DoNotOptimize(gmtime_s(&tm, &t));
#endif  // _WIN32
  }
}
BENCHMARK(BM_Time_ToDateTimeUTC_Libc);

//
// FromUnixMicros
//

void BM_Time_FromUnixMicros(benchmark::State& state) {
  int i = 0;
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(abslx::FromUnixMicros(i));
    ++i;
  }
}
BENCHMARK(BM_Time_FromUnixMicros);

void BM_Time_ToUnixNanos(benchmark::State& state) {
  const abslx::Time t = abslx::UnixEpoch() + abslx::Seconds(123);
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(ToUnixNanos(t));
  }
}
BENCHMARK(BM_Time_ToUnixNanos);

void BM_Time_ToUnixMicros(benchmark::State& state) {
  const abslx::Time t = abslx::UnixEpoch() + abslx::Seconds(123);
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(ToUnixMicros(t));
  }
}
BENCHMARK(BM_Time_ToUnixMicros);

void BM_Time_ToUnixMillis(benchmark::State& state) {
  const abslx::Time t = abslx::UnixEpoch() + abslx::Seconds(123);
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(ToUnixMillis(t));
  }
}
BENCHMARK(BM_Time_ToUnixMillis);

void BM_Time_ToUnixSeconds(benchmark::State& state) {
  const abslx::Time t = abslx::UnixEpoch() + abslx::Seconds(123);
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(abslx::ToUnixSeconds(t));
  }
}
BENCHMARK(BM_Time_ToUnixSeconds);

//
// FromCivil
//
// In each "FromCivil" benchmark we switch between two YMDhms values
// separated by at least one transition in order to defeat any internal
// caching of previous results (e.g., see time_local_hint_).
//
// The "UTC" variants use UTC instead of the Google/local time zone.
// The "Day0" variants require normalization of the day of month.
//

void BM_Time_FromCivil_Absl(benchmark::State& state) {
  const abslx::TimeZone tz =
      abslx::time_internal::LoadTimeZone("America/Los_Angeles");
  int i = 0;
  while (state.KeepRunning()) {
    if ((i & 1) == 0) {
      abslx::FromCivil(abslx::CivilSecond(2014, 12, 18, 20, 16, 18), tz);
    } else {
      abslx::FromCivil(abslx::CivilSecond(2013, 11, 15, 18, 30, 27), tz);
    }
    ++i;
  }
}
BENCHMARK(BM_Time_FromCivil_Absl);

void BM_Time_FromCivil_Libc(benchmark::State& state) {
  // No timezone support, so just use localtime.
  int i = 0;
  while (state.KeepRunning()) {
    struct tm tm;
    if ((i & 1) == 0) {
      tm.tm_year = 2014 - 1900;
      tm.tm_mon = 12 - 1;
      tm.tm_mday = 18;
      tm.tm_hour = 20;
      tm.tm_min = 16;
      tm.tm_sec = 18;
    } else {
      tm.tm_year = 2013 - 1900;
      tm.tm_mon = 11 - 1;
      tm.tm_mday = 15;
      tm.tm_hour = 18;
      tm.tm_min = 30;
      tm.tm_sec = 27;
    }
    tm.tm_isdst = -1;
    mktime(&tm);
    ++i;
  }
}
BENCHMARK(BM_Time_FromCivil_Libc);

void BM_Time_FromCivilUTC_Absl(benchmark::State& state) {
  const abslx::TimeZone tz = abslx::UTCTimeZone();
  while (state.KeepRunning()) {
    abslx::FromCivil(abslx::CivilSecond(2014, 12, 18, 20, 16, 18), tz);
  }
}
BENCHMARK(BM_Time_FromCivilUTC_Absl);

void BM_Time_FromCivilDay0_Absl(benchmark::State& state) {
  const abslx::TimeZone tz =
      abslx::time_internal::LoadTimeZone("America/Los_Angeles");
  int i = 0;
  while (state.KeepRunning()) {
    if ((i & 1) == 0) {
      abslx::FromCivil(abslx::CivilSecond(2014, 12, 0, 20, 16, 18), tz);
    } else {
      abslx::FromCivil(abslx::CivilSecond(2013, 11, 0, 18, 30, 27), tz);
    }
    ++i;
  }
}
BENCHMARK(BM_Time_FromCivilDay0_Absl);

void BM_Time_FromCivilDay0_Libc(benchmark::State& state) {
  // No timezone support, so just use localtime.
  int i = 0;
  while (state.KeepRunning()) {
    struct tm tm;
    if ((i & 1) == 0) {
      tm.tm_year = 2014 - 1900;
      tm.tm_mon = 12 - 1;
      tm.tm_mday = 0;
      tm.tm_hour = 20;
      tm.tm_min = 16;
      tm.tm_sec = 18;
    } else {
      tm.tm_year = 2013 - 1900;
      tm.tm_mon = 11 - 1;
      tm.tm_mday = 0;
      tm.tm_hour = 18;
      tm.tm_min = 30;
      tm.tm_sec = 27;
    }
    tm.tm_isdst = -1;
    mktime(&tm);
    ++i;
  }
}
BENCHMARK(BM_Time_FromCivilDay0_Libc);

//
// To/FromTimespec
//

void BM_Time_ToTimespec(benchmark::State& state) {
  abslx::Time now = abslx::Now();
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(abslx::ToTimespec(now));
  }
}
BENCHMARK(BM_Time_ToTimespec);

void BM_Time_FromTimespec(benchmark::State& state) {
  timespec ts = abslx::ToTimespec(abslx::Now());
  while (state.KeepRunning()) {
    if (++ts.tv_nsec == 1000 * 1000 * 1000) {
      ++ts.tv_sec;
      ts.tv_nsec = 0;
    }
    benchmark::DoNotOptimize(abslx::TimeFromTimespec(ts));
  }
}
BENCHMARK(BM_Time_FromTimespec);

//
// Comparison with InfiniteFuture/Past
//

void BM_Time_InfiniteFuture(benchmark::State& state) {
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(abslx::InfiniteFuture());
  }
}
BENCHMARK(BM_Time_InfiniteFuture);

void BM_Time_InfinitePast(benchmark::State& state) {
  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(abslx::InfinitePast());
  }
}
BENCHMARK(BM_Time_InfinitePast);

}  // namespace
