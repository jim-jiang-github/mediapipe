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

#ifndef MEDIAPIPE_DEPS_STATUS_BUILDER_H_
#define MEDIAPIPE_DEPS_STATUS_BUILDER_H_

#include <memory>
#include <sstream>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/source_location.h"
#include "mediapipe/framework/deps/status.h"

namespace mediapipe {

class ABSL_MUST_USE_RESULT StatusBuilder {
 public:
  StatusBuilder(const StatusBuilder& sb);
  StatusBuilder& operator=(const StatusBuilder& sb);

  StatusBuilder(StatusBuilder&&) = default;
  StatusBuilder& operator=(StatusBuilder&&) = default;

  // Creates a `StatusBuilder` based on an original status.  If logging is
  // enabled, it will use `location` as the location from which the log message
  // occurs.  A typical user will call this with `MEDIAPIPE_LOC`.
  StatusBuilder(const abslx::Status& original_status,
                mediapipe::source_location location)
      : impl_(original_status.ok()
                  ? nullptr
                  : std::make_unique<Impl>(original_status, location)) {}

  StatusBuilder(abslx::Status&& original_status,
                mediapipe::source_location location)
      : impl_(original_status.ok()
                  ? nullptr
                  : std::make_unique<Impl>(std::move(original_status),
                                           location)) {}

  // Creates a `StatusBuilder` from a mediapipe status code.  If logging is
  // enabled, it will use `location` as the location from which the log message
  // occurs.  A typical user will call this with `MEDIAPIPE_LOC`.
  StatusBuilder(abslx::StatusCode code, mediapipe::source_location location)
      : impl_(code == abslx::StatusCode::kOk
                  ? nullptr
                  : std::make_unique<Impl>(abslx::Status(code, ""), location)) {}

  bool ok() const { return !impl_; }

  StatusBuilder& SetAppend() &;
  StatusBuilder&& SetAppend() &&;

  StatusBuilder& SetPrepend() &;
  StatusBuilder&& SetPrepend() &&;

  StatusBuilder& SetNoLogging() &;
  StatusBuilder&& SetNoLogging() &&;

  template <typename T>
  StatusBuilder& operator<<(const T& msg) & {
    if (!impl_) return *this;
    impl_->stream << msg;
    return *this;
  }

  template <typename T>
  StatusBuilder&& operator<<(const T& msg) && {
    return std::move(*this << msg);
  }

  operator Status() const&;
  operator Status() &&;

  abslx::Status JoinMessageToStatus();

 private:
  struct Impl {
    // Specifies how to join the error message in the original status and any
    // additional message that has been streamed into the builder.
    enum class MessageJoinStyle {
      kAnnotate,
      kAppend,
      kPrepend,
    };

    Impl(const abslx::Status& status, mediapipe::source_location location);
    Impl(abslx::Status&& status, mediapipe::source_location location);
    Impl(const Impl&);
    Impl& operator=(const Impl&);

    abslx::Status JoinMessageToStatus();

    // The status that the result will be based on.
    abslx::Status status;
    // The source location to record if this file is logged.
    mediapipe::source_location location;
    // Logging disabled if true.
    bool no_logging = false;
    // The additional messages added with `<<`.  This is nullptr when status_ is
    // ok.
    std::ostringstream stream;
    // Specifies how to join the message in `status_` and `stream_`.
    MessageJoinStyle join_style = MessageJoinStyle::kAnnotate;
  };

  // Internal store of data for the class.  An invariant of the class is that
  // this is null when the original status is okay, and not-null otherwise.
  std::unique_ptr<Impl> impl_;
};

inline StatusBuilder AlreadyExistsErrorBuilder(
    mediapipe::source_location location) {
  return StatusBuilder(abslx::StatusCode::kAlreadyExists, location);
}

inline StatusBuilder FailedPreconditionErrorBuilder(
    mediapipe::source_location location) {
  return StatusBuilder(abslx::StatusCode::kFailedPrecondition, location);
}

inline StatusBuilder InternalErrorBuilder(mediapipe::source_location location) {
  return StatusBuilder(abslx::StatusCode::kInternal, location);
}

inline StatusBuilder InvalidArgumentErrorBuilder(
    mediapipe::source_location location) {
  return StatusBuilder(abslx::StatusCode::kInvalidArgument, location);
}

inline StatusBuilder NotFoundErrorBuilder(mediapipe::source_location location) {
  return StatusBuilder(abslx::StatusCode::kNotFound, location);
}

inline StatusBuilder UnavailableErrorBuilder(
    mediapipe::source_location location) {
  return StatusBuilder(abslx::StatusCode::kUnavailable, location);
}

inline StatusBuilder UnimplementedErrorBuilder(
    mediapipe::source_location location) {
  return StatusBuilder(abslx::StatusCode::kUnimplemented, location);
}

inline StatusBuilder UnknownErrorBuilder(mediapipe::source_location location) {
  return StatusBuilder(abslx::StatusCode::kUnknown, location);
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_DEPS_STATUS_BUILDER_H_
