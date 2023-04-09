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

#ifndef MEDIAPIPE_DEPS_CANONICAL_ERRORS_H_
#define MEDIAPIPE_DEPS_CANONICAL_ERRORS_H_

#include "mediapipe/framework/deps/status.h"

namespace mediapipe {

// Each of the functions below creates a canonical error with the given
// message. The error code of the returned status object matches the name of
// the function.
inline abslx::Status AlreadyExistsError(abslx::string_view message) {
  return abslx::Status(abslx::StatusCode::kAlreadyExists, message);
}

inline abslx::Status CancelledError() {
  return abslx::Status(abslx::StatusCode::kCancelled, "");
}

inline abslx::Status CancelledError(abslx::string_view message) {
  return abslx::Status(abslx::StatusCode::kCancelled, message);
}

inline abslx::Status InternalError(abslx::string_view message) {
  return abslx::Status(abslx::StatusCode::kInternal, message);
}

inline abslx::Status InvalidArgumentError(abslx::string_view message) {
  return abslx::Status(abslx::StatusCode::kInvalidArgument, message);
}

inline abslx::Status FailedPreconditionError(abslx::string_view message) {
  return abslx::Status(abslx::StatusCode::kFailedPrecondition, message);
}

inline abslx::Status NotFoundError(abslx::string_view message) {
  return abslx::Status(abslx::StatusCode::kNotFound, message);
}

inline abslx::Status OutOfRangeError(abslx::string_view message) {
  return abslx::Status(abslx::StatusCode::kOutOfRange, message);
}

inline abslx::Status PermissionDeniedError(abslx::string_view message) {
  return abslx::Status(abslx::StatusCode::kPermissionDenied, message);
}

inline abslx::Status UnimplementedError(abslx::string_view message) {
  return abslx::Status(abslx::StatusCode::kUnimplemented, message);
}

inline abslx::Status UnknownError(abslx::string_view message) {
  return abslx::Status(abslx::StatusCode::kUnknown, message);
}

inline abslx::Status UnavailableError(abslx::string_view message) {
  return abslx::Status(abslx::StatusCode::kUnavailable, message);
}

inline bool IsCancelled(const abslx::Status& status) {
  return status.code() == abslx::StatusCode::kCancelled;
}

inline bool IsNotFound(const abslx::Status& status) {
  return status.code() == abslx::StatusCode::kNotFound;
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_DEPS_CANONICAL_ERRORS_H_
