// Copyright 2019 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "absl/status/status.h"

#include <cassert>

#include "absl/base/internal/raw_logging.h"
#include "absl/debugging/stacktrace.h"
#include "absl/debugging/symbolize.h"
#include "absl/status/status_payload_printer.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"

namespace abslx {
ABSL_NAMESPACE_BEGIN

std::string StatusCodeToString(StatusCode code) {
  switch (code) {
    case StatusCode::kOk:
      return "OK";
    case StatusCode::kCancelled:
      return "CANCELLED";
    case StatusCode::kUnknown:
      return "UNKNOWN";
    case StatusCode::kInvalidArgument:
      return "INVALID_ARGUMENT";
    case StatusCode::kDeadlineExceeded:
      return "DEADLINE_EXCEEDED";
    case StatusCode::kNotFound:
      return "NOT_FOUND";
    case StatusCode::kAlreadyExists:
      return "ALREADY_EXISTS";
    case StatusCode::kPermissionDenied:
      return "PERMISSION_DENIED";
    case StatusCode::kUnauthenticated:
      return "UNAUTHENTICATED";
    case StatusCode::kResourceExhausted:
      return "RESOURCE_EXHAUSTED";
    case StatusCode::kFailedPrecondition:
      return "FAILED_PRECONDITION";
    case StatusCode::kAborted:
      return "ABORTED";
    case StatusCode::kOutOfRange:
      return "OUT_OF_RANGE";
    case StatusCode::kUnimplemented:
      return "UNIMPLEMENTED";
    case StatusCode::kInternal:
      return "INTERNAL";
    case StatusCode::kUnavailable:
      return "UNAVAILABLE";
    case StatusCode::kDataLoss:
      return "DATA_LOSS";
    default:
      return "";
  }
}

std::ostream& operator<<(std::ostream& os, StatusCode code) {
  return os << StatusCodeToString(code);
}

namespace status_internal {

static int FindPayloadIndexByUrl(const Payloads* payloads,
                                 abslx::string_view type_url) {
  if (payloads == nullptr) return -1;

  for (size_t i = 0; i < payloads->size(); ++i) {
    if ((*payloads)[i].type_url == type_url) return i;
  }

  return -1;
}

// Convert canonical code to a value known to this binary.
abslx::StatusCode MapToLocalCode(int value) {
  abslx::StatusCode code = static_cast<abslx::StatusCode>(value);
  switch (code) {
    case abslx::StatusCode::kOk:
    case abslx::StatusCode::kCancelled:
    case abslx::StatusCode::kUnknown:
    case abslx::StatusCode::kInvalidArgument:
    case abslx::StatusCode::kDeadlineExceeded:
    case abslx::StatusCode::kNotFound:
    case abslx::StatusCode::kAlreadyExists:
    case abslx::StatusCode::kPermissionDenied:
    case abslx::StatusCode::kResourceExhausted:
    case abslx::StatusCode::kFailedPrecondition:
    case abslx::StatusCode::kAborted:
    case abslx::StatusCode::kOutOfRange:
    case abslx::StatusCode::kUnimplemented:
    case abslx::StatusCode::kInternal:
    case abslx::StatusCode::kUnavailable:
    case abslx::StatusCode::kDataLoss:
    case abslx::StatusCode::kUnauthenticated:
      return code;
    default:
      return abslx::StatusCode::kUnknown;
  }
}
}  // namespace status_internal

abslx::optional<abslx::Cord> Status::GetPayload(
    abslx::string_view type_url) const {
  const auto* payloads = GetPayloads();
  int index = status_internal::FindPayloadIndexByUrl(payloads, type_url);
  if (index != -1) return (*payloads)[index].payload;

  return abslx::nullopt;
}

void Status::SetPayload(abslx::string_view type_url, abslx::Cord payload) {
  if (ok()) return;

  PrepareToModify();

  status_internal::StatusRep* rep = RepToPointer(rep_);
  if (!rep->payloads) {
    rep->payloads = abslx::make_unique<status_internal::Payloads>();
  }

  int index =
      status_internal::FindPayloadIndexByUrl(rep->payloads.get(), type_url);
  if (index != -1) {
    (*rep->payloads)[index].payload = std::move(payload);
    return;
  }

  rep->payloads->push_back({std::string(type_url), std::move(payload)});
}

bool Status::ErasePayload(abslx::string_view type_url) {
  int index = status_internal::FindPayloadIndexByUrl(GetPayloads(), type_url);
  if (index != -1) {
    PrepareToModify();
    GetPayloads()->erase(GetPayloads()->begin() + index);
    if (GetPayloads()->empty() && message().empty()) {
      // Special case: If this can be represented inlined, it MUST be
      // inlined (EqualsSlow depends on this behavior).
      StatusCode c = static_cast<StatusCode>(raw_code());
      Unref(rep_);
      rep_ = CodeToInlinedRep(c);
    }
    return true;
  }

  return false;
}

void Status::ForEachPayload(
    const std::function<void(abslx::string_view, const abslx::Cord&)>& visitor)
    const {
  if (auto* payloads = GetPayloads()) {
    bool in_reverse =
        payloads->size() > 1 && reinterpret_cast<uintptr_t>(payloads) % 13 > 6;

    for (size_t index = 0; index < payloads->size(); ++index) {
      const auto& elem =
          (*payloads)[in_reverse ? payloads->size() - 1 - index : index];

#ifdef NDEBUG
      visitor(elem.type_url, elem.payload);
#else
      // In debug mode invalidate the type url to prevent users from relying on
      // this string lifetime.

      // NOLINTNEXTLINE intentional extra conversion to force temporary.
      visitor(std::string(elem.type_url), elem.payload);
#endif  // NDEBUG
    }
  }
}

const std::string* Status::EmptyString() {
  static std::string* empty_string = new std::string();
  return empty_string;
}

constexpr const char Status::kMovedFromString[];

const std::string* Status::MovedFromString() {
  static std::string* moved_from_string = new std::string(kMovedFromString);
  return moved_from_string;
}

void Status::UnrefNonInlined(uintptr_t rep) {
  status_internal::StatusRep* r = RepToPointer(rep);
  // Fast path: if ref==1, there is no need for a RefCountDec (since
  // this is the only reference and therefore no other thread is
  // allowed to be mucking with r).
  if (r->ref.load(std::memory_order_acquire) == 1 ||
      r->ref.fetch_sub(1, std::memory_order_acq_rel) - 1 == 0) {
    delete r;
  }
}

Status::Status(abslx::StatusCode code, abslx::string_view msg)
    : rep_(CodeToInlinedRep(code)) {
  if (code != abslx::StatusCode::kOk && !msg.empty()) {
    rep_ = PointerToRep(new status_internal::StatusRep(code, msg, nullptr));
  }
}

int Status::raw_code() const {
  if (IsInlined(rep_)) {
    return static_cast<int>(InlinedRepToCode(rep_));
  }
  status_internal::StatusRep* rep = RepToPointer(rep_);
  return static_cast<int>(rep->code);
}

abslx::StatusCode Status::code() const {
  return status_internal::MapToLocalCode(raw_code());
}

void Status::PrepareToModify() {
  ABSL_RAW_CHECK(!ok(), "PrepareToModify shouldn't be called on OK status.");
  if (IsInlined(rep_)) {
    rep_ = PointerToRep(new status_internal::StatusRep(
        static_cast<abslx::StatusCode>(raw_code()), abslx::string_view(),
        nullptr));
    return;
  }

  uintptr_t rep_i = rep_;
  status_internal::StatusRep* rep = RepToPointer(rep_);
  if (rep->ref.load(std::memory_order_acquire) != 1) {
    std::unique_ptr<status_internal::Payloads> payloads;
    if (rep->payloads) {
      payloads = abslx::make_unique<status_internal::Payloads>(*rep->payloads);
    }
    status_internal::StatusRep* const new_rep = new status_internal::StatusRep(
        rep->code, message(), std::move(payloads));
    rep_ = PointerToRep(new_rep);
    UnrefNonInlined(rep_i);
  }
}

bool Status::EqualsSlow(const abslx::Status& a, const abslx::Status& b) {
  if (IsInlined(a.rep_) != IsInlined(b.rep_)) return false;
  if (a.message() != b.message()) return false;
  if (a.raw_code() != b.raw_code()) return false;
  if (a.GetPayloads() == b.GetPayloads()) return true;

  const status_internal::Payloads no_payloads;
  const status_internal::Payloads* larger_payloads =
      a.GetPayloads() ? a.GetPayloads() : &no_payloads;
  const status_internal::Payloads* smaller_payloads =
      b.GetPayloads() ? b.GetPayloads() : &no_payloads;
  if (larger_payloads->size() < smaller_payloads->size()) {
    std::swap(larger_payloads, smaller_payloads);
  }
  if ((larger_payloads->size() - smaller_payloads->size()) > 1) return false;
  // Payloads can be ordered differently, so we can't just compare payload
  // vectors.
  for (const auto& payload : *larger_payloads) {

    bool found = false;
    for (const auto& other_payload : *smaller_payloads) {
      if (payload.type_url == other_payload.type_url) {
        if (payload.payload != other_payload.payload) {
          return false;
        }
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

std::string Status::ToStringSlow(StatusToStringMode mode) const {
  std::string text;
  abslx::StrAppend(&text, abslx::StatusCodeToString(code()), ": ", message());

  const bool with_payload = (mode & StatusToStringMode::kWithPayload) ==
                      StatusToStringMode::kWithPayload;

  if (with_payload) {
    status_internal::StatusPayloadPrinter printer =
        status_internal::GetStatusPayloadPrinter();
    this->ForEachPayload([&](abslx::string_view type_url,
                             const abslx::Cord& payload) {
      abslx::optional<std::string> result;
      if (printer) result = printer(type_url, payload);
      abslx::StrAppend(
          &text, " [", type_url, "='",
          result.has_value() ? *result : abslx::CHexEscape(std::string(payload)),
          "']");
    });
  }

  return text;
}

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

Status AbortedError(abslx::string_view message) {
  return Status(abslx::StatusCode::kAborted, message);
}

Status AlreadyExistsError(abslx::string_view message) {
  return Status(abslx::StatusCode::kAlreadyExists, message);
}

Status CancelledError(abslx::string_view message) {
  return Status(abslx::StatusCode::kCancelled, message);
}

Status DataLossError(abslx::string_view message) {
  return Status(abslx::StatusCode::kDataLoss, message);
}

Status DeadlineExceededError(abslx::string_view message) {
  return Status(abslx::StatusCode::kDeadlineExceeded, message);
}

Status FailedPreconditionError(abslx::string_view message) {
  return Status(abslx::StatusCode::kFailedPrecondition, message);
}

Status InternalError(abslx::string_view message) {
  return Status(abslx::StatusCode::kInternal, message);
}

Status InvalidArgumentError(abslx::string_view message) {
  return Status(abslx::StatusCode::kInvalidArgument, message);
}

Status NotFoundError(abslx::string_view message) {
  return Status(abslx::StatusCode::kNotFound, message);
}

Status OutOfRangeError(abslx::string_view message) {
  return Status(abslx::StatusCode::kOutOfRange, message);
}

Status PermissionDeniedError(abslx::string_view message) {
  return Status(abslx::StatusCode::kPermissionDenied, message);
}

Status ResourceExhaustedError(abslx::string_view message) {
  return Status(abslx::StatusCode::kResourceExhausted, message);
}

Status UnauthenticatedError(abslx::string_view message) {
  return Status(abslx::StatusCode::kUnauthenticated, message);
}

Status UnavailableError(abslx::string_view message) {
  return Status(abslx::StatusCode::kUnavailable, message);
}

Status UnimplementedError(abslx::string_view message) {
  return Status(abslx::StatusCode::kUnimplemented, message);
}

Status UnknownError(abslx::string_view message) {
  return Status(abslx::StatusCode::kUnknown, message);
}

bool IsAborted(const Status& status) {
  return status.code() == abslx::StatusCode::kAborted;
}

bool IsAlreadyExists(const Status& status) {
  return status.code() == abslx::StatusCode::kAlreadyExists;
}

bool IsCancelled(const Status& status) {
  return status.code() == abslx::StatusCode::kCancelled;
}

bool IsDataLoss(const Status& status) {
  return status.code() == abslx::StatusCode::kDataLoss;
}

bool IsDeadlineExceeded(const Status& status) {
  return status.code() == abslx::StatusCode::kDeadlineExceeded;
}

bool IsFailedPrecondition(const Status& status) {
  return status.code() == abslx::StatusCode::kFailedPrecondition;
}

bool IsInternal(const Status& status) {
  return status.code() == abslx::StatusCode::kInternal;
}

bool IsInvalidArgument(const Status& status) {
  return status.code() == abslx::StatusCode::kInvalidArgument;
}

bool IsNotFound(const Status& status) {
  return status.code() == abslx::StatusCode::kNotFound;
}

bool IsOutOfRange(const Status& status) {
  return status.code() == abslx::StatusCode::kOutOfRange;
}

bool IsPermissionDenied(const Status& status) {
  return status.code() == abslx::StatusCode::kPermissionDenied;
}

bool IsResourceExhausted(const Status& status) {
  return status.code() == abslx::StatusCode::kResourceExhausted;
}

bool IsUnauthenticated(const Status& status) {
  return status.code() == abslx::StatusCode::kUnauthenticated;
}

bool IsUnavailable(const Status& status) {
  return status.code() == abslx::StatusCode::kUnavailable;
}

bool IsUnimplemented(const Status& status) {
  return status.code() == abslx::StatusCode::kUnimplemented;
}

bool IsUnknown(const Status& status) {
  return status.code() == abslx::StatusCode::kUnknown;
}

ABSL_NAMESPACE_END
}  // namespace abslx
