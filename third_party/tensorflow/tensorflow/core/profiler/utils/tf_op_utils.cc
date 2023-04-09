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

#include "tensorflow/core/profiler/utils/tf_op_utils.h"

#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace profiler {
namespace {

const abslx::string_view kIterator = "Iterator";
const abslx::string_view kSeparator = "::";
constexpr char kNameScopeSeparator = '/';
constexpr char kOpNameSuffixSeparator = '_';

bool IsInteger(abslx::string_view str) {
  int64_t unused;
  return abslx::SimpleAtoi(str, &unused);
}

// Returns an op type derived from an op name.
abslx::string_view DeriveOpType(abslx::string_view full_op_name) {
  // Use the op name without name scopes and suffix as an op type. A full op
  // name consists of name scopes, an op type, and optionally a numeric suffix
  // (e.g., model/layer/MatMul_1).
  std::vector<abslx::string_view> name_scopes_and_op_name =
      abslx::StrSplit(full_op_name, kNameScopeSeparator);
  abslx::string_view op_name = name_scopes_and_op_name.back();
  std::vector<abslx::string_view> op_type_and_maybe_suffix =
      abslx::StrSplit(op_name, kOpNameSuffixSeparator);
  abslx::string_view maybe_suffix = op_type_and_maybe_suffix.back();
  abslx::string_view op_type = op_name;
  if (IsInteger(maybe_suffix)) {
    // NOTE: assuming a numeric suffix is not part of an op type while
    // technically it is allowed.
    op_type = op_name.substr(0, op_name.size() - maybe_suffix.size() - 1);
  }
  return op_type;
}

}  // namespace

const abslx::string_view kUnknownOp = "";  // op types are non-empty strings
const abslx::string_view kDatasetOp = "Dataset";
const abslx::string_view kMemcpyHToDOp = "MemcpyHToD";
const abslx::string_view kMemcpyDToHOp = "MemcpyDToH";
const abslx::string_view kMemcpyDToDOp = "MemcpyDToD";
const abslx::string_view kMemcpyHToHOp = "MemcpyHToH";

bool IsTfOpName(abslx::string_view op_name) {
  // TODO(b/177602927): Confirm the naming convention with the TF team.
  static const LazyRE2 kTfOpNameRegEx = {"[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*"};
  return RE2::FullMatch(op_name, *kTfOpNameRegEx);
}

bool IsTfOpType(abslx::string_view op_type) {
  static const LazyRE2 kTfOpTypeRegEx = {"[A-Z_][a-zA-Z0-9_]*"};
  return RE2::FullMatch(op_type, *kTfOpTypeRegEx);
}

bool IsJaxOpType(abslx::string_view op_type) {
  static const LazyRE2 kJaxOpTypeRegEx = {"[a-z_][a-z0-9_]*"};
  return RE2::FullMatch(op_type, *kJaxOpTypeRegEx);
}

bool IsJaxOpNameAndType(abslx::string_view op_name, abslx::string_view op_type) {
  if (op_name.empty() || !IsJaxOpType(op_type)) return false;
  std::vector<abslx::string_view> split_result =
      abslx::StrSplit(op_name, kNameScopeSeparator);
  return abslx::StrContains(split_result.back(), op_type);
}

TfOp ParseTfOpFullname(abslx::string_view tf_op_fullname) {
  // TF Op names have the format "name:type".
  TfOp tf_op = {Category::kUnknown, tf_op_fullname, kUnknownOp};
  std::vector<abslx::string_view> parts =
      abslx::StrSplit(tf_op_fullname, abslx::MaxSplits(':', 1));
  if (parts.size() != 2) {
    // GPU-related Ops that need to be tracked.
    if (abslx::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYHToD")) {
      tf_op.category = Category::kMemcpyHToD;
      tf_op.type = kMemcpyHToDOp;
    } else if (abslx::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYDToH")) {
      tf_op.category = Category::kMemcpyDToH;
      tf_op.type = kMemcpyDToHOp;
    } else if (abslx::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYDToD")) {
      tf_op.category = Category::kMemcpyDToD;
      tf_op.type = kMemcpyDToDOp;
    } else if (abslx::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYHToH")) {
      tf_op.category = Category::kMemcpyHToH;
      tf_op.type = kMemcpyHToHOp;
    }
    // TODO(ckluk): Include the corresponding Ops on TPU.
  } else if (parts[0] == kIterator) {
    // Dataset Op names (e.g., Iterator::Batch::Map::TFRecord) do not follow the
    // format of TF Op names. But we still want to capture them for
    // input-pipeline analysis.
    tf_op.category = Category::kTfData;
    tf_op.type = kDatasetOp;
  } else if (IsTfOpType(parts[1]) && IsTfOpName(parts[0])) {
    tf_op = {Category::kTensorFlow, parts[0], parts[1]};
  } else if (IsJaxOpType(parts[1])) {
    tf_op = {Category::kJax, parts[0], parts[1]};
  } else if (parts[1].empty()) {
    tf_op = {Category::kTensorFlow, parts[0], DeriveOpType(parts[0])};
  }
  return tf_op;
}

std::vector<abslx::string_view> ParseTfNameScopes(abslx::string_view tf_op_name) {
  std::vector<abslx::string_view> name_scopes =
      abslx::StrSplit(tf_op_name, kNameScopeSeparator);
  // The last element is an op name not TF name scope.
  if (!name_scopes.empty()) name_scopes.pop_back();
  return name_scopes;
}

std::vector<abslx::string_view> ParseTfNameScopes(const TfOp& tf_op) {
  return ParseTfNameScopes(tf_op.name);
}

std::string TfOpEventName(const TfOp& tf_op) {
  std::string event_name;
  if (tf_op.category == Category::kUnknown) {
    // Some TraceMe names contain trailing whitespace, remove it.
    event_name = std::string(abslx::StripTrailingAsciiWhitespace(tf_op.name));
  } else if (tf_op.category == Category::kTfData) {
    event_name = DatasetOpEventName(tf_op.name);
  } else {
    event_name = std::string(tf_op.type);
  }
  return event_name;
}

std::string TfOpEventName(abslx::string_view tf_op_fullname) {
  return TfOpEventName(ParseTfOpFullname(tf_op_fullname));
}

std::string DatasetOpEventName(abslx::string_view full_name) {
  std::vector<abslx::string_view> split_result =
      abslx::StrSplit(full_name, kSeparator);
  return abslx::StrCat(kIterator, kSeparator, split_result.back());
}

std::string IteratorName(abslx::string_view full_name) {
  std::vector<abslx::string_view> split_result =
      abslx::StrSplit(full_name, kSeparator);
  return std::string(split_result.back());
}

std::vector<abslx::string_view> ParseTensorShapes(
    abslx::string_view tensor_shapes) {
  abslx::ConsumePrefix(&tensor_shapes, "(");
  abslx::ConsumeSuffix(&tensor_shapes, ")");
  return abslx::StrSplit(tensor_shapes, ';');
}

}  // namespace profiler
}  // namespace tensorflow
