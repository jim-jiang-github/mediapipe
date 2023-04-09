// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MEDIAPIPE_FRAMEWORK_TOOL_STATUS_UTIL_H_
#define MEDIAPIPE_FRAMEWORK_TOOL_STATUS_UTIL_H_

#include <string>
#include <vector>

#include "absl/base/macros.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace tool {

// Return a status which signals that an action should stop.  For example,
// a source Calculator is done producing output (and Process() should not
// be called on it again).  When returned from a non-source Calculator
// it signals that the graph should be cancelled (which is handled by
// closing all source Calculators and waiting for the graph to finish).
abslx::Status StatusStop();

// Return a status which signals an invalid initial condition (for
// example an InputSidePacket does not include all necessary fields).
ABSL_DEPRECATED("Use abslx::InvalidArgumentError(error_message) instead.")
abslx::Status StatusInvalid(const std::string& error_message);

// Return a status which signals that something unexpectedly failed.
ABSL_DEPRECATED("Use abslx::UnknownError(error_message) instead.")
abslx::Status StatusFail(const std::string& error_message);

// Prefixes the given string to the error message in status.
// This function should be considered internal to the framework.
// TODO Replace usage of AddStatusPrefix with util::Annotate().
abslx::Status AddStatusPrefix(const std::string& prefix,
                             const abslx::Status& status);

// Combine a vector of abslx::Status into a single composite status.
// If statuses is empty or all statuses are OK then abslx::OkStatus()
// will be returned.
// This function should be considered internal to the framework.
// TODO Move this function to somewhere with less visibility.
abslx::Status CombinedStatus(const std::string& general_comment,
                            const std::vector<abslx::Status>& statuses);

}  // namespace tool
}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_TOOL_STATUS_UTIL_H_
