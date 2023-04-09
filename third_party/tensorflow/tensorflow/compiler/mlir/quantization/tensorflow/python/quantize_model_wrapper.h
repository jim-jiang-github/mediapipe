/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_QUANTIZE_MODEL_WRAPPER_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_QUANTIZE_MODEL_WRAPPER_H_

#include <pybind11/stl.h>

#include <string>

#include "absl/strings/string_view.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"

namespace tensorflow {
namespace quantization {

PyObject* QuantizeQatModel(abslx::string_view saved_model_path,
                           abslx::string_view exported_names_str,
                           abslx::string_view tags,
                           abslx::string_view quant_opts_serialized);

PyObject* QuantizePtqDynamicRange(abslx::string_view saved_model_path,
                                  abslx::string_view exported_names_str,
                                  abslx::string_view tags,
                                  abslx::string_view quant_opts_serialized);

PyObject* QuantizePtqModelPreCalibration(abslx::string_view saved_model_path,
                                         abslx::string_view exported_names_str,
                                         abslx::string_view tags);

PyObject* QuantizePtqModelPostCalibration(
    abslx::string_view saved_model_path, abslx::string_view exported_names_str,
    abslx::string_view tags, abslx::string_view quant_opts_serialized);

void ClearCollectedInformationFromCalibrator();

void ClearDataFromCalibrator(abslx::string_view id);

float GetMinFromCalibrator(abslx::string_view id);

float GetMaxFromCalibrator(abslx::string_view id);

}  // namespace quantization
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PYTHON_QUANTIZE_MODEL_WRAPPER_H_
