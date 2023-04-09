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

#include "tensorflow/core/tpu/tpu_embedding_errors.h"

#include <string>

#include "absl/strings/match.h"

namespace tensorflow::tpu {

Status AppendTpuEmbeddingErrorPayload(Status obj) {
  if (obj.ok()) {
    return OkStatus();
  } else {
    const std::string error_message =
        abslx::StrCat(kTpuEmbeddingErrorMessage, ". ", obj.error_message());
    Status status(obj.code(), error_message);
    TPUEmbeddingError error_payload;
    status.SetPayload(kTpuEmbeddingErrorUrl, error_payload.SerializeAsString());
    return status;
  }
}

bool HasTpuEmbeddingErrorPayload(const Status& status) {
  return status.GetPayload(kTpuEmbeddingErrorUrl).has_value();
}

bool HasTpuEmbeddingErrorMessage(const Status& status) {
  return abslx::StrContains(status.error_message(), kTpuEmbeddingErrorMessage);
}

}  // namespace tensorflow::tpu
