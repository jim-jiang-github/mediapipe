/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_ELEMENTWISE_TEST_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_ELEMENTWISE_TEST_UTIL_H_

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"

namespace tflite {
namespace gpu {

abslx::Status AbsTest(TestExecutionEnvironment* env);
abslx::Status CosTest(TestExecutionEnvironment* env);
abslx::Status CopyTest(TestExecutionEnvironment* env);
abslx::Status EluTest(TestExecutionEnvironment* env);
abslx::Status ExpTest(TestExecutionEnvironment* env);
abslx::Status FloorTest(TestExecutionEnvironment* env);
abslx::Status FloorDivTest(TestExecutionEnvironment* env);
abslx::Status FloorModTest(TestExecutionEnvironment* env);
abslx::Status HardSwishTest(TestExecutionEnvironment* env);
abslx::Status LogTest(TestExecutionEnvironment* env);
abslx::Status NegTest(TestExecutionEnvironment* env);
abslx::Status RsqrtTest(TestExecutionEnvironment* env);
abslx::Status SigmoidTest(TestExecutionEnvironment* env);
abslx::Status SinTest(TestExecutionEnvironment* env);
abslx::Status SqrtTest(TestExecutionEnvironment* env);
abslx::Status SquareTest(TestExecutionEnvironment* env);
abslx::Status TanhTest(TestExecutionEnvironment* env);
abslx::Status SubTest(TestExecutionEnvironment* env);
abslx::Status SquaredDiffTest(TestExecutionEnvironment* env);
abslx::Status DivTest(TestExecutionEnvironment* env);
abslx::Status PowTest(TestExecutionEnvironment* env);
abslx::Status AddTest(TestExecutionEnvironment* env);
abslx::Status MaximumTest(TestExecutionEnvironment* env);
abslx::Status MaximumWithScalarTest(TestExecutionEnvironment* env);
abslx::Status MaximumWithConstantLinearTensorTest(TestExecutionEnvironment* env);
abslx::Status MaximumWithConstantHWCTensorTest(TestExecutionEnvironment* env);
abslx::Status MaximumWithConstantHWCTensorBroadcastChannelsTest(
    TestExecutionEnvironment* env);
abslx::Status MinimumTest(TestExecutionEnvironment* env);
abslx::Status MinimumWithScalarTest(TestExecutionEnvironment* env);
abslx::Status MulTest(TestExecutionEnvironment* env);
abslx::Status MulBroadcastHWTest(TestExecutionEnvironment* env);
abslx::Status MulBroadcastChannelsTest(TestExecutionEnvironment* env);
abslx::Status SubWithScalarAtFirstPositionTest(TestExecutionEnvironment* env);
abslx::Status LessTest(TestExecutionEnvironment* env);
abslx::Status LessEqualTest(TestExecutionEnvironment* env);
abslx::Status GreaterTest(TestExecutionEnvironment* env);
abslx::Status GreaterEqualTest(TestExecutionEnvironment* env);
abslx::Status EqualTest(TestExecutionEnvironment* env);
abslx::Status NotEqualTest(TestExecutionEnvironment* env);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_ELEMENTWISE_TEST_UTIL_H_
