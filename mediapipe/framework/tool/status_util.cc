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

#include "mediapipe/framework/tool/status_util.h"

#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace mediapipe {
namespace tool {

abslx::Status StatusInvalid(const std::string& message) {
  return abslx::Status(abslx::StatusCode::kInvalidArgument, message);
}

abslx::Status StatusFail(const std::string& message) {
  return abslx::Status(abslx::StatusCode::kUnknown, message);
}

abslx::Status StatusStop() {
  return abslx::Status(abslx::StatusCode::kOutOfRange,
                      "mediapipe::tool::StatusStop()");
}

abslx::Status AddStatusPrefix(const std::string& prefix,
                             const abslx::Status& status) {
  return abslx::Status(status.code(), abslx::StrCat(prefix, status.message()));
}

abslx::Status CombinedStatus(const std::string& general_comment,
                            const std::vector<abslx::Status>& statuses) {
  // The final error code is abslx::StatusCode::kUnknown if not all
  // the error codes are the same.  Otherwise it is the same error code
  // as all of the (non-OK) statuses.  If statuses is empty or they are
  // all OK, then abslx::OkStatus() is returned.
  abslx::StatusCode error_code = abslx::StatusCode::kOk;
  std::vector<std::string> errors;
  for (const abslx::Status& status : statuses) {
    if (!status.ok()) {
      errors.emplace_back(status.message());
      if (error_code == abslx::StatusCode::kOk) {
        error_code = status.code();
      } else if (error_code != status.code()) {
        error_code = abslx::StatusCode::kUnknown;
      }
    }
  }
  if (error_code == StatusCode::kOk) return OkStatus();
  Status combined;
  combined = abslx::Status(
      error_code,
      abslx::StrCat(general_comment, "\n", abslx::StrJoin(errors, "\n")));
  return combined;
}

}  // namespace tool
}  // namespace mediapipe
