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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_TF_OP_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_TF_OP_UTILS_H_

#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace profiler {

// Special op types.
TF_CONST_INIT extern const abslx::string_view kUnknownOp;
TF_CONST_INIT extern const abslx::string_view kDatasetOp;
TF_CONST_INIT extern const abslx::string_view kMemcpyHToDOp;
TF_CONST_INIT extern const abslx::string_view kMemcpyDToHOp;
TF_CONST_INIT extern const abslx::string_view kMemcpyDToDOp;
TF_CONST_INIT extern const abslx::string_view kMemcpyHToHOp;

enum class Category {
  kUnknown,
  kTensorFlow,
  kJax,
  kTfData,
  kMemcpyHToD,
  kMemcpyDToH,
  kMemcpyDToD,
  kMemcpyHToH,
};

// Breaks a TensorFlow op fullname into name and type.
struct TfOp {
  Category category = Category::kUnknown;
  abslx::string_view name;
  abslx::string_view type;
};
TfOp ParseTfOpFullname(abslx::string_view tf_op_fullname);

// Returns a vector of TF name scopes extracted from a TF op name.
std::vector<abslx::string_view> ParseTfNameScopes(abslx::string_view tf_op_name);
std::vector<abslx::string_view> ParseTfNameScopes(const TfOp& tf_op);

// Trace event name for TF ops is the op type so they have the same color in
// trace viewer.
std::string TfOpEventName(const TfOp& tf_op);
std::string TfOpEventName(abslx::string_view tf_op_fullname);

// Trace event name for dataset ops.
std::string DatasetOpEventName(abslx::string_view full_name);

// Returns the iterator name without prefix and parent iterator names.
std::string IteratorName(abslx::string_view full_name);

// Returns true if the given name is a TensorFlow Dataset Op.
inline bool IsDatasetOp(abslx::string_view tf_op_type) {
  return tf_op_type == kDatasetOp;
}
inline bool IsDatasetOp(const TfOp& tf_op) {
  return tf_op.category == Category::kTfData;
}

// Returns true if the given name is a TensorFlow Infeed Enqueue Op.
// See: tensorflow/core/tpu/kernels/infeed_ops.h
inline bool IsInfeedEnqueueOp(abslx::string_view tf_op_type) {
  return abslx::StartsWith(tf_op_type, "InfeedEnqueue");
}
inline bool IsInfeedEnqueueOp(const TfOp& tf_op) {
  return tf_op.category == Category::kTensorFlow &&
         IsInfeedEnqueueOp(tf_op.type);
}

// Returns true if the given op has XlaSendToHost/XlaRecvFromHost in fullname.
inline bool IsOutsideCompilationOp(abslx::string_view tf_op_fullname) {
  if (abslx::EndsWith(tf_op_fullname, ":XlaSendToHost")) return true;
  if (abslx::EndsWith(tf_op_fullname, ":XlaRecvFromHost")) return true;
  return false;
}

// Returns true if the given op is for outside compilation.
inline bool IsOutsideCompilationOp(abslx::string_view tf_op_fullname,
                                   abslx::string_view hlo_expression) {
  if (IsOutsideCompilationOp(tf_op_fullname)) return true;
  if (abslx::StrContains(hlo_expression, "send-done") &&
      abslx::StrContains(hlo_expression, "is_host_transfer=true"))
    return true;
  return false;
}

// Returns true if the given name is a TensorFlow embedding op.
inline bool IsEmbeddingOp(abslx::string_view tf_op_fullname) {
  return abslx::StrContains(tf_op_fullname, "Embedding");
}

// Returns true if the given op is for copying data from host to device.
inline bool IsMemcpyHToDOp(abslx::string_view tf_op_type) {
  return tf_op_type == kMemcpyHToDOp;
}
inline bool IsMemcpyHToDOp(const TfOp& tf_op) {
  return tf_op.category == Category::kMemcpyHToD;
}

// Returns true if the given op is for copying data from device to host.
inline bool IsMemcpyDToHOp(const TfOp& tf_op) {
  return tf_op.category == Category::kMemcpyDToH;
}

// Returns true if the given op is for copying data from device to device.
inline bool IsMemcpyDToDOp(const TfOp& tf_op) {
  return tf_op.category == Category::kMemcpyDToD;
}

// Returns true if the given op is for copying data from host to host.
inline bool IsMemcpyHToHOp(const TfOp& tf_op) {
  return tf_op.category == Category::kMemcpyHToH;
}

// Splits a string of tensor shapes in "(shape1;shape2;...)" format, i.e.,
// delimited by '(' and ')' and separated by ';', into the individual shapes.
std::vector<abslx::string_view> ParseTensorShapes(
    abslx::string_view tensor_shapes);

// Returns true if the given string matches OpDef.name pattern.
bool IsTfOpName(abslx::string_view op_name);

// Returns true if the given string matches NodeDef.name pattern.
bool IsTfOpType(abslx::string_view op_type);

// Returns true if the given string matches JAX pattern.
bool IsJaxOpType(abslx::string_view op_type);

// Returns true if the given strings match JAX pattern.
bool IsJaxOpNameAndType(abslx::string_view op_name, abslx::string_view op_type);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_TF_OP_UTILS_H_
