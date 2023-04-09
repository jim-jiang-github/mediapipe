/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/slow_operation_alarm.h"

#include <functional>
#include <iterator>
#include <list>
#include <memory>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/numeric/bits.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/env.h"

namespace xla {
namespace {

abslx::Mutex mu(abslx::kConstInit);
abslx::CondVar* ready;
abslx::once_flag init_flag;
std::list<SlowOperationAlarm*>* outstanding_alarms ABSL_PT_GUARDED_BY(mu) =
    nullptr;

}  // namespace

void SlowOperationAlarm::AlarmLoop() {
  while (true) {
    abslx::MutexLock lock(&mu);

    // Fire any alarms which are ready.
    abslx::Time now = abslx::Now();
    for (auto it = outstanding_alarms->begin();
         it != outstanding_alarms->end();) {
      auto next = std::next(it);
      auto* alarm = *it;
      // Fire the alarm if applicable.
      if (alarm->deadline() <= now) {
        outstanding_alarms->erase(it);
        const int64_t count =
            alarm->counter() == nullptr ? 0 : alarm->counter()->fetch_add(1);
        // If the alarm has a counter, only fire if the count is a power of 2.
        if (count == 0 || abslx::has_single_bit<uint64_t>(count) == 0) {
          alarm->fired_.store(true);
          // We fire alarms with LOG(ERROR) because otherwise it might not show
          // up without --logtostderr.
          LOG(ERROR) << alarm->msg();
        }
      }
      it = next;
    }

    auto next_alarm = abslx::c_min_element(
        *outstanding_alarms,
        [](const SlowOperationAlarm* a, const SlowOperationAlarm* b) {
          return a->deadline() < b->deadline();
        });
    const abslx::Time deadline = next_alarm != outstanding_alarms->end()
                                    ? (*next_alarm)->deadline()
                                    : abslx::InfiniteFuture();

    ready->WaitWithDeadline(&mu, deadline);
  }
}

void SlowOperationAlarm::ScheduleAlarm(SlowOperationAlarm* alarm) {
  abslx::call_once(init_flag, [] {
    ready = new abslx::CondVar();
    outstanding_alarms = new std::list<SlowOperationAlarm*>();
    (void)tensorflow::Env::Default()->StartThread(
        tensorflow::ThreadOptions(), "SlowOperationAlarm", [] { AlarmLoop(); });
  });

  abslx::MutexLock lock(&mu);
  outstanding_alarms->push_back(alarm);
  ready->Signal();
}

void SlowOperationAlarm::UnscheduleAlarm(const SlowOperationAlarm* alarm) {
  abslx::MutexLock lock(&mu);
  CHECK(outstanding_alarms != nullptr);
  auto it = abslx::c_find(*outstanding_alarms, alarm);
  if (it != outstanding_alarms->end()) {
    outstanding_alarms->erase(it);
  }
}
SlowOperationAlarm::SlowOperationAlarm(
    abslx::Duration timeout, std::string msg,
    std::atomic<int64_t>* counter /*=nullptr*/,
    abslx::string_view context /*=""*/)
    : SlowOperationAlarm(
          timeout,                                 //
          [msg = std::move(msg)] { return msg; },  //
          counter, std::move(context)) {}

SlowOperationAlarm::SlowOperationAlarm(
    abslx::Duration timeout, std::function<std::string()> msg_fn,
    std::atomic<int64_t>* counter /*=nullptr*/,
    abslx::string_view context /*=""*/)
    : start_(abslx::Now()),
      deadline_(start_ + timeout),
      context_(std::move(context)),
      msg_fn_(std::move(msg_fn)),
      counter_(counter) {
  ScheduleAlarm(this);
}

SlowOperationAlarm::~SlowOperationAlarm() {
  UnscheduleAlarm(this);

  abslx::Time now = abslx::Now();
  if (deadline() <= now) {
    abslx::Duration duration = now - start_;
    if (context_.empty()) {
      LOG(ERROR) << "The operation took " << abslx::FormatDuration(duration)
                 << "\n"
                 << msg_fn_();
    } else {
      LOG(ERROR) << "[" << context_ << "] The operation took "
                 << abslx::FormatDuration(duration) << "\n"
                 << msg_fn_();
    }
  }
}

std::unique_ptr<SlowOperationAlarm> SlowCompilationAlarm(
    abslx::string_view context) {
  // Pass a counter to these alarms so they only log once every power-of-two
  // occurrences.
  static auto* counter = new std::atomic<int64_t>(0);

  const char* separator = "\n********************************";

  std::string context_msg;
  if (!context.empty()) {
    context_msg = abslx::StrCat("[", context, "] ");
  }

#if NDEBUG
  return std::make_unique<SlowOperationAlarm>(
      abslx::Duration(abslx::Minutes(2)),
      abslx::StrCat(
          separator, "\n", context_msg,
          "Very slow compile?  If you want to file a bug, run with envvar "
          "XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.",
          separator),
      counter);
#else
  return std::make_unique<SlowOperationAlarm>(
      abslx::Duration(abslx::Seconds(10)),
      abslx::StrCat(
          separator, "\n", context_msg,
          "Slow compile?  XLA was built without compiler optimizations, "
          "which can be slow.  Try rebuilding with -c opt.",
          separator),
      counter);
#endif
}

}  // namespace xla
