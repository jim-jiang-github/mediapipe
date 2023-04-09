/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/reporter.h"

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/str_util.h"

namespace tensorflow {

TestReportFile::TestReportFile(const string& fname, const string& test_name)
    : closed_(true), fname_(fname), test_name_(test_name) {}

Status TestReportFile::Append(const string& content) {
  if (closed_) return OkStatus();
  return log_file_->Append(content);
}

Status TestReportFile::Close() {
  if (closed_) return OkStatus();
  closed_ = true;
  return log_file_->Close();
}

Status TestReportFile::Initialize() {
  if (fname_.empty()) {
    return OkStatus();
  }
  string mangled_fname = strings::StrCat(
      fname_, abslx::StrJoin(str_util::Split(test_name_, '/'), "__"));
  Env* env = Env::Default();
  if (env->FileExists(mangled_fname).ok()) {
    return errors::InvalidArgument(
        "Cannot create TestReportFile, file exists: ", mangled_fname);
  }
  TF_RETURN_IF_ERROR(env->NewWritableFile(mangled_fname, &log_file_));
  TF_RETURN_IF_ERROR(log_file_->Flush());

  closed_ = false;
  return OkStatus();
}

TestReporter::TestReporter(const string& fname, const string& test_name)
    : report_file_(fname, test_name) {
  benchmark_entry_.set_name(test_name);
}

Status TestReporter::Close() {
  if (report_file_.IsClosed()) return OkStatus();

  BenchmarkEntries entries;
  *entries.add_entry() = benchmark_entry_;
  TF_RETURN_IF_ERROR(report_file_.Append(entries.SerializeAsString()));
  benchmark_entry_.Clear();

  return report_file_.Close();
}

Status TestReporter::Benchmark(int64_t iters, double cpu_time, double wall_time,
                               double throughput) {
  if (report_file_.IsClosed()) return OkStatus();
  benchmark_entry_.set_iters(iters);
  benchmark_entry_.set_cpu_time(cpu_time / iters);
  benchmark_entry_.set_wall_time(wall_time / iters);
  benchmark_entry_.set_throughput(throughput);
  return OkStatus();
}

Status TestReporter::SetProperty(const string& name, const string& value) {
  if (report_file_.IsClosed()) return OkStatus();
  (*benchmark_entry_.mutable_extras())[name].set_string_value(value);
  return OkStatus();
}

Status TestReporter::SetProperty(const string& name, double value) {
  if (report_file_.IsClosed()) return OkStatus();
  (*benchmark_entry_.mutable_extras())[name].set_double_value(value);
  return OkStatus();
}

Status TestReporter::AddMetric(const string& name, double value) {
  if (report_file_.IsClosed()) return OkStatus();
  auto* metric = benchmark_entry_.add_metrics();
  metric->set_name(name);
  metric->set_value(value);
  return OkStatus();
}

Status TestReporter::Initialize() { return report_file_.Initialize(); }

}  // namespace tensorflow
