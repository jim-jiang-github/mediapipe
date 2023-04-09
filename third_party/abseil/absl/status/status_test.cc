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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"

namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Optional;
using ::testing::UnorderedElementsAreArray;

TEST(StatusCode, InsertionOperator) {
  const abslx::StatusCode code = abslx::StatusCode::kUnknown;
  std::ostringstream oss;
  oss << code;
  EXPECT_EQ(oss.str(), abslx::StatusCodeToString(code));
}

// This structure holds the details for testing a single error code,
// its creator, and its classifier.
struct ErrorTest {
  abslx::StatusCode code;
  using Creator = abslx::Status (*)(
      abslx::string_view
  );
  using Classifier = bool (*)(const abslx::Status&);
  Creator creator;
  Classifier classifier;
};

constexpr ErrorTest kErrorTests[]{
    {abslx::StatusCode::kCancelled, abslx::CancelledError, abslx::IsCancelled},
    {abslx::StatusCode::kUnknown, abslx::UnknownError, abslx::IsUnknown},
    {abslx::StatusCode::kInvalidArgument, abslx::InvalidArgumentError,
     abslx::IsInvalidArgument},
    {abslx::StatusCode::kDeadlineExceeded, abslx::DeadlineExceededError,
     abslx::IsDeadlineExceeded},
    {abslx::StatusCode::kNotFound, abslx::NotFoundError, abslx::IsNotFound},
    {abslx::StatusCode::kAlreadyExists, abslx::AlreadyExistsError,
     abslx::IsAlreadyExists},
    {abslx::StatusCode::kPermissionDenied, abslx::PermissionDeniedError,
     abslx::IsPermissionDenied},
    {abslx::StatusCode::kResourceExhausted, abslx::ResourceExhaustedError,
     abslx::IsResourceExhausted},
    {abslx::StatusCode::kFailedPrecondition, abslx::FailedPreconditionError,
     abslx::IsFailedPrecondition},
    {abslx::StatusCode::kAborted, abslx::AbortedError, abslx::IsAborted},
    {abslx::StatusCode::kOutOfRange, abslx::OutOfRangeError, abslx::IsOutOfRange},
    {abslx::StatusCode::kUnimplemented, abslx::UnimplementedError,
     abslx::IsUnimplemented},
    {abslx::StatusCode::kInternal, abslx::InternalError, abslx::IsInternal},
    {abslx::StatusCode::kUnavailable, abslx::UnavailableError,
     abslx::IsUnavailable},
    {abslx::StatusCode::kDataLoss, abslx::DataLossError, abslx::IsDataLoss},
    {abslx::StatusCode::kUnauthenticated, abslx::UnauthenticatedError,
     abslx::IsUnauthenticated},
};

TEST(Status, CreateAndClassify) {
  for (const auto& test : kErrorTests) {
    SCOPED_TRACE(abslx::StatusCodeToString(test.code));

    // Ensure that the creator does, in fact, create status objects with the
    // expected error code and message.
    std::string message =
        abslx::StrCat("error code ", test.code, " test message");
    abslx::Status status = test.creator(
        message
    );
    EXPECT_EQ(test.code, status.code());
    EXPECT_EQ(message, status.message());

    // Ensure that the classifier returns true for a status produced by the
    // creator.
    EXPECT_TRUE(test.classifier(status));

    // Ensure that the classifier returns false for status with a different
    // code.
    for (const auto& other : kErrorTests) {
      if (other.code != test.code) {
        EXPECT_FALSE(test.classifier(abslx::Status(other.code, "")))
            << " other.code = " << other.code;
      }
    }
  }
}

TEST(Status, DefaultConstructor) {
  abslx::Status status;
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(abslx::StatusCode::kOk, status.code());
  EXPECT_EQ("", status.message());
}

TEST(Status, OkStatus) {
  abslx::Status status = abslx::OkStatus();
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(abslx::StatusCode::kOk, status.code());
  EXPECT_EQ("", status.message());
}

TEST(Status, ConstructorWithCodeMessage) {
  {
    abslx::Status status(abslx::StatusCode::kCancelled, "");
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(abslx::StatusCode::kCancelled, status.code());
    EXPECT_EQ("", status.message());
  }
  {
    abslx::Status status(abslx::StatusCode::kInternal, "message");
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(abslx::StatusCode::kInternal, status.code());
    EXPECT_EQ("message", status.message());
  }
}

TEST(Status, ConstructOutOfRangeCode) {
  const int kRawCode = 9999;
  abslx::Status status(static_cast<abslx::StatusCode>(kRawCode), "");
  EXPECT_EQ(abslx::StatusCode::kUnknown, status.code());
  EXPECT_EQ(kRawCode, status.raw_code());
}

constexpr char kUrl1[] = "url.payload.1";
constexpr char kUrl2[] = "url.payload.2";
constexpr char kUrl3[] = "url.payload.3";
constexpr char kUrl4[] = "url.payload.xx";

constexpr char kPayload1[] = "aaaaa";
constexpr char kPayload2[] = "bbbbb";
constexpr char kPayload3[] = "ccccc";

using PayloadsVec = std::vector<std::pair<std::string, abslx::Cord>>;

TEST(Status, TestGetSetPayload) {
  abslx::Status ok_status = abslx::OkStatus();
  ok_status.SetPayload(kUrl1, abslx::Cord(kPayload1));
  ok_status.SetPayload(kUrl2, abslx::Cord(kPayload2));

  EXPECT_FALSE(ok_status.GetPayload(kUrl1));
  EXPECT_FALSE(ok_status.GetPayload(kUrl2));

  abslx::Status bad_status(abslx::StatusCode::kInternal, "fail");
  bad_status.SetPayload(kUrl1, abslx::Cord(kPayload1));
  bad_status.SetPayload(kUrl2, abslx::Cord(kPayload2));

  EXPECT_THAT(bad_status.GetPayload(kUrl1), Optional(Eq(kPayload1)));
  EXPECT_THAT(bad_status.GetPayload(kUrl2), Optional(Eq(kPayload2)));

  EXPECT_FALSE(bad_status.GetPayload(kUrl3));

  bad_status.SetPayload(kUrl1, abslx::Cord(kPayload3));
  EXPECT_THAT(bad_status.GetPayload(kUrl1), Optional(Eq(kPayload3)));

  // Testing dynamically generated type_url
  bad_status.SetPayload(abslx::StrCat(kUrl1, ".1"), abslx::Cord(kPayload1));
  EXPECT_THAT(bad_status.GetPayload(abslx::StrCat(kUrl1, ".1")),
              Optional(Eq(kPayload1)));
}

TEST(Status, TestErasePayload) {
  abslx::Status bad_status(abslx::StatusCode::kInternal, "fail");
  bad_status.SetPayload(kUrl1, abslx::Cord(kPayload1));
  bad_status.SetPayload(kUrl2, abslx::Cord(kPayload2));
  bad_status.SetPayload(kUrl3, abslx::Cord(kPayload3));

  EXPECT_FALSE(bad_status.ErasePayload(kUrl4));

  EXPECT_TRUE(bad_status.GetPayload(kUrl2));
  EXPECT_TRUE(bad_status.ErasePayload(kUrl2));
  EXPECT_FALSE(bad_status.GetPayload(kUrl2));
  EXPECT_FALSE(bad_status.ErasePayload(kUrl2));

  EXPECT_TRUE(bad_status.ErasePayload(kUrl1));
  EXPECT_TRUE(bad_status.ErasePayload(kUrl3));

  bad_status.SetPayload(kUrl1, abslx::Cord(kPayload1));
  EXPECT_TRUE(bad_status.ErasePayload(kUrl1));
}

TEST(Status, TestComparePayloads) {
  abslx::Status bad_status1(abslx::StatusCode::kInternal, "fail");
  bad_status1.SetPayload(kUrl1, abslx::Cord(kPayload1));
  bad_status1.SetPayload(kUrl2, abslx::Cord(kPayload2));
  bad_status1.SetPayload(kUrl3, abslx::Cord(kPayload3));

  abslx::Status bad_status2(abslx::StatusCode::kInternal, "fail");
  bad_status2.SetPayload(kUrl2, abslx::Cord(kPayload2));
  bad_status2.SetPayload(kUrl3, abslx::Cord(kPayload3));
  bad_status2.SetPayload(kUrl1, abslx::Cord(kPayload1));

  EXPECT_EQ(bad_status1, bad_status2);
}

TEST(Status, TestComparePayloadsAfterErase) {
  abslx::Status payload_status(abslx::StatusCode::kInternal, "");
  payload_status.SetPayload(kUrl1, abslx::Cord(kPayload1));
  payload_status.SetPayload(kUrl2, abslx::Cord(kPayload2));

  abslx::Status empty_status(abslx::StatusCode::kInternal, "");

  // Different payloads, not equal
  EXPECT_NE(payload_status, empty_status);
  EXPECT_TRUE(payload_status.ErasePayload(kUrl1));

  // Still Different payloads, still not equal.
  EXPECT_NE(payload_status, empty_status);
  EXPECT_TRUE(payload_status.ErasePayload(kUrl2));

  // Both empty payloads, should be equal
  EXPECT_EQ(payload_status, empty_status);
}

PayloadsVec AllVisitedPayloads(const abslx::Status& s) {
  PayloadsVec result;

  s.ForEachPayload([&](abslx::string_view type_url, const abslx::Cord& payload) {
    result.push_back(std::make_pair(std::string(type_url), payload));
  });

  return result;
}

TEST(Status, TestForEachPayload) {
  abslx::Status bad_status(abslx::StatusCode::kInternal, "fail");
  bad_status.SetPayload(kUrl1, abslx::Cord(kPayload1));
  bad_status.SetPayload(kUrl2, abslx::Cord(kPayload2));
  bad_status.SetPayload(kUrl3, abslx::Cord(kPayload3));

  int count = 0;

  bad_status.ForEachPayload(
      [&count](abslx::string_view, const abslx::Cord&) { ++count; });

  EXPECT_EQ(count, 3);

  PayloadsVec expected_payloads = {{kUrl1, abslx::Cord(kPayload1)},
                                   {kUrl2, abslx::Cord(kPayload2)},
                                   {kUrl3, abslx::Cord(kPayload3)}};

  // Test that we visit all the payloads in the status.
  PayloadsVec visited_payloads = AllVisitedPayloads(bad_status);
  EXPECT_THAT(visited_payloads, UnorderedElementsAreArray(expected_payloads));

  // Test that visitation order is not consistent between run.
  std::vector<abslx::Status> scratch;
  while (true) {
    scratch.emplace_back(abslx::StatusCode::kInternal, "fail");

    scratch.back().SetPayload(kUrl1, abslx::Cord(kPayload1));
    scratch.back().SetPayload(kUrl2, abslx::Cord(kPayload2));
    scratch.back().SetPayload(kUrl3, abslx::Cord(kPayload3));

    if (AllVisitedPayloads(scratch.back()) != visited_payloads) {
      break;
    }
  }
}

TEST(Status, ToString) {
  abslx::Status s(abslx::StatusCode::kInternal, "fail");
  EXPECT_EQ("INTERNAL: fail", s.ToString());
  s.SetPayload("foo", abslx::Cord("bar"));
  EXPECT_EQ("INTERNAL: fail [foo='bar']", s.ToString());
  s.SetPayload("bar", abslx::Cord("\377"));
  EXPECT_THAT(s.ToString(),
              AllOf(HasSubstr("INTERNAL: fail"), HasSubstr("[foo='bar']"),
                    HasSubstr("[bar='\\xff']")));
}

TEST(Status, ToStringMode) {
  abslx::Status s(abslx::StatusCode::kInternal, "fail");
  s.SetPayload("foo", abslx::Cord("bar"));
  s.SetPayload("bar", abslx::Cord("\377"));

  EXPECT_EQ("INTERNAL: fail",
            s.ToString(abslx::StatusToStringMode::kWithNoExtraData));

  EXPECT_THAT(s.ToString(abslx::StatusToStringMode::kWithPayload),
              AllOf(HasSubstr("INTERNAL: fail"), HasSubstr("[foo='bar']"),
                    HasSubstr("[bar='\\xff']")));

  EXPECT_THAT(s.ToString(abslx::StatusToStringMode::kWithEverything),
              AllOf(HasSubstr("INTERNAL: fail"), HasSubstr("[foo='bar']"),
                    HasSubstr("[bar='\\xff']")));

  EXPECT_THAT(s.ToString(~abslx::StatusToStringMode::kWithPayload),
              AllOf(HasSubstr("INTERNAL: fail"), Not(HasSubstr("[foo='bar']")),
                    Not(HasSubstr("[bar='\\xff']"))));
}

abslx::Status EraseAndReturn(const abslx::Status& base) {
  abslx::Status copy = base;
  EXPECT_TRUE(copy.ErasePayload(kUrl1));
  return copy;
}

TEST(Status, CopyOnWriteForErasePayload) {
  {
    abslx::Status base(abslx::StatusCode::kInvalidArgument, "fail");
    base.SetPayload(kUrl1, abslx::Cord(kPayload1));
    EXPECT_TRUE(base.GetPayload(kUrl1).has_value());
    abslx::Status copy = EraseAndReturn(base);
    EXPECT_TRUE(base.GetPayload(kUrl1).has_value());
    EXPECT_FALSE(copy.GetPayload(kUrl1).has_value());
  }
  {
    abslx::Status base(abslx::StatusCode::kInvalidArgument, "fail");
    base.SetPayload(kUrl1, abslx::Cord(kPayload1));
    abslx::Status copy = base;

    EXPECT_TRUE(base.GetPayload(kUrl1).has_value());
    EXPECT_TRUE(copy.GetPayload(kUrl1).has_value());

    EXPECT_TRUE(base.ErasePayload(kUrl1));

    EXPECT_FALSE(base.GetPayload(kUrl1).has_value());
    EXPECT_TRUE(copy.GetPayload(kUrl1).has_value());
  }
}

TEST(Status, CopyConstructor) {
  {
    abslx::Status status;
    abslx::Status copy(status);
    EXPECT_EQ(copy, status);
  }
  {
    abslx::Status status(abslx::StatusCode::kInvalidArgument, "message");
    abslx::Status copy(status);
    EXPECT_EQ(copy, status);
  }
  {
    abslx::Status status(abslx::StatusCode::kInvalidArgument, "message");
    status.SetPayload(kUrl1, abslx::Cord(kPayload1));
    abslx::Status copy(status);
    EXPECT_EQ(copy, status);
  }
}

TEST(Status, CopyAssignment) {
  abslx::Status assignee;
  {
    abslx::Status status;
    assignee = status;
    EXPECT_EQ(assignee, status);
  }
  {
    abslx::Status status(abslx::StatusCode::kInvalidArgument, "message");
    assignee = status;
    EXPECT_EQ(assignee, status);
  }
  {
    abslx::Status status(abslx::StatusCode::kInvalidArgument, "message");
    status.SetPayload(kUrl1, abslx::Cord(kPayload1));
    assignee = status;
    EXPECT_EQ(assignee, status);
  }
}

TEST(Status, CopyAssignmentIsNotRef) {
  const abslx::Status status_orig(abslx::StatusCode::kInvalidArgument, "message");
  abslx::Status status_copy = status_orig;
  EXPECT_EQ(status_orig, status_copy);
  status_copy.SetPayload(kUrl1, abslx::Cord(kPayload1));
  EXPECT_NE(status_orig, status_copy);
}

TEST(Status, MoveConstructor) {
  {
    abslx::Status status;
    abslx::Status copy(abslx::Status{});
    EXPECT_EQ(copy, status);
  }
  {
    abslx::Status status(abslx::StatusCode::kInvalidArgument, "message");
    abslx::Status copy(
        abslx::Status(abslx::StatusCode::kInvalidArgument, "message"));
    EXPECT_EQ(copy, status);
  }
  {
    abslx::Status status(abslx::StatusCode::kInvalidArgument, "message");
    status.SetPayload(kUrl1, abslx::Cord(kPayload1));
    abslx::Status copy1(status);
    abslx::Status copy2(std::move(status));
    EXPECT_EQ(copy1, copy2);
  }
}

TEST(Status, MoveAssignment) {
  abslx::Status assignee;
  {
    abslx::Status status;
    assignee = abslx::Status();
    EXPECT_EQ(assignee, status);
  }
  {
    abslx::Status status(abslx::StatusCode::kInvalidArgument, "message");
    assignee = abslx::Status(abslx::StatusCode::kInvalidArgument, "message");
    EXPECT_EQ(assignee, status);
  }
  {
    abslx::Status status(abslx::StatusCode::kInvalidArgument, "message");
    status.SetPayload(kUrl1, abslx::Cord(kPayload1));
    abslx::Status copy(status);
    assignee = std::move(status);
    EXPECT_EQ(assignee, copy);
  }
  {
    abslx::Status status(abslx::StatusCode::kInvalidArgument, "message");
    abslx::Status copy(status);
    status = static_cast<abslx::Status&&>(status);
    EXPECT_EQ(status, copy);
  }
}

TEST(Status, Update) {
  abslx::Status s;
  s.Update(abslx::OkStatus());
  EXPECT_TRUE(s.ok());
  const abslx::Status a(abslx::StatusCode::kCancelled, "message");
  s.Update(a);
  EXPECT_EQ(s, a);
  const abslx::Status b(abslx::StatusCode::kInternal, "other message");
  s.Update(b);
  EXPECT_EQ(s, a);
  s.Update(abslx::OkStatus());
  EXPECT_EQ(s, a);
  EXPECT_FALSE(s.ok());
}

TEST(Status, Equality) {
  abslx::Status ok;
  abslx::Status no_payload = abslx::CancelledError("no payload");
  abslx::Status one_payload = abslx::InvalidArgumentError("one payload");
  one_payload.SetPayload(kUrl1, abslx::Cord(kPayload1));
  abslx::Status two_payloads = one_payload;
  two_payloads.SetPayload(kUrl2, abslx::Cord(kPayload2));
  const std::array<abslx::Status, 4> status_arr = {ok, no_payload, one_payload,
                                                  two_payloads};
  for (int i = 0; i < status_arr.size(); i++) {
    for (int j = 0; j < status_arr.size(); j++) {
      if (i == j) {
        EXPECT_TRUE(status_arr[i] == status_arr[j]);
        EXPECT_FALSE(status_arr[i] != status_arr[j]);
      } else {
        EXPECT_TRUE(status_arr[i] != status_arr[j]);
        EXPECT_FALSE(status_arr[i] == status_arr[j]);
      }
    }
  }
}

TEST(Status, Swap) {
  auto test_swap = [](const abslx::Status& s1, const abslx::Status& s2) {
    abslx::Status copy1 = s1, copy2 = s2;
    swap(copy1, copy2);
    EXPECT_EQ(copy1, s2);
    EXPECT_EQ(copy2, s1);
  };
  const abslx::Status ok;
  const abslx::Status no_payload(abslx::StatusCode::kAlreadyExists, "no payload");
  abslx::Status with_payload(abslx::StatusCode::kInternal, "with payload");
  with_payload.SetPayload(kUrl1, abslx::Cord(kPayload1));
  test_swap(ok, no_payload);
  test_swap(no_payload, ok);
  test_swap(ok, with_payload);
  test_swap(with_payload, ok);
  test_swap(no_payload, with_payload);
  test_swap(with_payload, no_payload);
}
}  // namespace
