//
// Copyright 2018 The Abseil Authors.
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
//

#ifndef ABSL_STRINGS_CORD_TEST_HELPERS_H_
#define ABSL_STRINGS_CORD_TEST_HELPERS_H_

#include "absl/strings/cord.h"

namespace abslx {
ABSL_NAMESPACE_BEGIN

// Creates a multi-segment Cord from an iterable container of strings.  The
// resulting Cord is guaranteed to have one segment for every string in the
// container.  This allows code to be unit tested with multi-segment Cord
// inputs.
//
// Example:
//
//   abslx::Cord c = abslx::MakeFragmentedCord({"A ", "fragmented ", "Cord"});
//   EXPECT_FALSE(c.GetFlat(&unused));
//
// The mechanism by which this Cord is created is an implementation detail.  Any
// implementation that produces a multi-segment Cord may produce a flat Cord in
// the future as new optimizations are added to the Cord class.
// MakeFragmentedCord will, however, always be updated to return a multi-segment
// Cord.
template <typename Container>
Cord MakeFragmentedCord(const Container& c) {
  Cord result;
  for (const auto& s : c) {
    auto* external = new std::string(s);
    Cord tmp = abslx::MakeCordFromExternal(
        *external, [external](abslx::string_view) { delete external; });
    tmp.Prepend(result);
    result = tmp;
  }
  return result;
}

inline Cord MakeFragmentedCord(std::initializer_list<abslx::string_view> list) {
  return MakeFragmentedCord<std::initializer_list<abslx::string_view>>(list);
}

ABSL_NAMESPACE_END
}  // namespace abslx

#endif  // ABSL_STRINGS_CORD_TEST_HELPERS_H_
