// Copyright 2020 The Abseil Authors.
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

#include "absl/strings/cord.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <iterator>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/casts.h"
#include "absl/base/config.h"
#include "absl/base/internal/endian.h"
#include "absl/base/internal/raw_logging.h"
#include "absl/base/macros.h"
#include "absl/container/fixed_array.h"
#include "absl/strings/cord_test_helpers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

typedef std::mt19937_64 RandomEngine;

static std::string RandomLowercaseString(RandomEngine* rng);
static std::string RandomLowercaseString(RandomEngine* rng, size_t length);

static int GetUniformRandomUpTo(RandomEngine* rng, int upper_bound) {
  if (upper_bound > 0) {
    std::uniform_int_distribution<int> uniform(0, upper_bound - 1);
    return uniform(*rng);
  } else {
    return 0;
  }
}

static size_t GetUniformRandomUpTo(RandomEngine* rng, size_t upper_bound) {
  if (upper_bound > 0) {
    std::uniform_int_distribution<size_t> uniform(0, upper_bound - 1);
    return uniform(*rng);
  } else {
    return 0;
  }
}

static int32_t GenerateSkewedRandom(RandomEngine* rng, int max_log) {
  const uint32_t base = (*rng)() % (max_log + 1);
  const uint32_t mask = ((base < 32) ? (1u << base) : 0u) - 1u;
  return (*rng)() & mask;
}

static std::string RandomLowercaseString(RandomEngine* rng) {
  int length;
  std::bernoulli_distribution one_in_1k(0.001);
  std::bernoulli_distribution one_in_10k(0.0001);
  // With low probability, make a large fragment
  if (one_in_10k(*rng)) {
    length = GetUniformRandomUpTo(rng, 1048576);
  } else if (one_in_1k(*rng)) {
    length = GetUniformRandomUpTo(rng, 10000);
  } else {
    length = GenerateSkewedRandom(rng, 10);
  }
  return RandomLowercaseString(rng, length);
}

static std::string RandomLowercaseString(RandomEngine* rng, size_t length) {
  std::string result(length, '\0');
  std::uniform_int_distribution<int> chars('a', 'z');
  std::generate(result.begin(), result.end(),
                [&]() { return static_cast<char>(chars(*rng)); });
  return result;
}

static void DoNothing(abslx::string_view /* data */, void* /* arg */) {}

static void DeleteExternalString(abslx::string_view data, void* arg) {
  std::string* s = reinterpret_cast<std::string*>(arg);
  EXPECT_EQ(data, *s);
  delete s;
}

// Add "s" to *dst via `MakeCordFromExternal`
static void AddExternalMemory(abslx::string_view s, abslx::Cord* dst) {
  std::string* str = new std::string(s.data(), s.size());
  dst->Append(abslx::MakeCordFromExternal(*str, [str](abslx::string_view data) {
    DeleteExternalString(data, str);
  }));
}

static void DumpGrowth() {
  abslx::Cord str;
  for (int i = 0; i < 1000; i++) {
    char c = 'a' + i % 26;
    str.Append(abslx::string_view(&c, 1));
  }
}

// Make a Cord with some number of fragments.  Return the size (in bytes)
// of the smallest fragment.
static size_t AppendWithFragments(const std::string& s, RandomEngine* rng,
                                  abslx::Cord* cord) {
  size_t j = 0;
  const size_t max_size = s.size() / 5;  // Make approx. 10 fragments
  size_t min_size = max_size;            // size of smallest fragment
  while (j < s.size()) {
    size_t N = 1 + GetUniformRandomUpTo(rng, max_size);
    if (N > (s.size() - j)) {
      N = s.size() - j;
    }
    if (N < min_size) {
      min_size = N;
    }

    std::bernoulli_distribution coin_flip(0.5);
    if (coin_flip(*rng)) {
      // Grow by adding an external-memory.
      AddExternalMemory(abslx::string_view(s.data() + j, N), cord);
    } else {
      cord->Append(abslx::string_view(s.data() + j, N));
    }
    j += N;
  }
  return min_size;
}

// Add an external memory that contains the specified std::string to cord
static void AddNewStringBlock(const std::string& str, abslx::Cord* dst) {
  char* data = new char[str.size()];
  memcpy(data, str.data(), str.size());
  dst->Append(abslx::MakeCordFromExternal(
      abslx::string_view(data, str.size()),
      [](abslx::string_view s) { delete[] s.data(); }));
}

// Make a Cord out of many different types of nodes.
static abslx::Cord MakeComposite() {
  abslx::Cord cord;
  cord.Append("the");
  AddExternalMemory(" quick brown", &cord);
  AddExternalMemory(" fox jumped", &cord);

  abslx::Cord full(" over");
  AddExternalMemory(" the lazy", &full);
  AddNewStringBlock(" dog slept the whole day away", &full);
  abslx::Cord substring = full.Subcord(0, 18);

  // Make substring long enough to defeat the copying fast path in Append.
  substring.Append(std::string(1000, '.'));
  cord.Append(substring);
  cord = cord.Subcord(0, cord.size() - 998);  // Remove most of extra junk

  return cord;
}

namespace abslx {
ABSL_NAMESPACE_BEGIN

class CordTestPeer {
 public:
  static void ForEachChunk(
      const Cord& c, abslx::FunctionRef<void(abslx::string_view)> callback) {
    c.ForEachChunk(callback);
  }

  static bool IsTree(const Cord& c) { return c.contents_.is_tree(); }

  static cord_internal::CordzInfo* GetCordzInfo(const Cord& c) {
    return c.contents_.cordz_info();
  }

  static Cord MakeSubstring(Cord src, size_t offset, size_t length) {
    Cord cord = src;
    ABSL_RAW_CHECK(cord.contents_.is_tree(), "Can not be inlined");
    auto* rep = new cord_internal::CordRepSubstring;
    rep->tag = cord_internal::SUBSTRING;
    rep->child = cord.contents_.tree();
    rep->start = offset;
    rep->length = length;
    cord.contents_.replace_tree(rep);
    return cord;
  }
};

ABSL_NAMESPACE_END
}  // namespace abslx

TEST(Cord, AllFlatSizes) {
  using abslx::strings_internal::CordTestAccess;

  for (size_t s = 0; s < CordTestAccess::MaxFlatLength(); s++) {
    // Make a string of length s.
    std::string src;
    while (src.size() < s) {
      src.push_back('a' + (src.size() % 26));
    }

    abslx::Cord dst(src);
    EXPECT_EQ(std::string(dst), src) << s;
  }
}

// We create a Cord at least 128GB in size using the fact that Cords can
// internally reference-count; thus the Cord is enormous without actually
// consuming very much memory.
TEST(GigabyteCord, FromExternal) {
  const size_t one_gig = 1024U * 1024U * 1024U;
  size_t max_size = 2 * one_gig;
  if (sizeof(max_size) > 4) max_size = 128 * one_gig;

  size_t length = 128 * 1024;
  char* data = new char[length];
  abslx::Cord from = abslx::MakeCordFromExternal(
      abslx::string_view(data, length),
      [](abslx::string_view sv) { delete[] sv.data(); });

  // This loop may seem odd due to its combination of exponential doubling of
  // size and incremental size increases.  We do it incrementally to be sure the
  // Cord will need rebalancing and will exercise code that, in the past, has
  // caused crashes in production.  We grow exponentially so that the code will
  // execute in a reasonable amount of time.
  abslx::Cord c;
  ABSL_RAW_LOG(INFO, "Made a Cord with %zu bytes!", c.size());
  c.Append(from);
  while (c.size() < max_size) {
    c.Append(c);
    c.Append(from);
    c.Append(from);
    c.Append(from);
    c.Append(from);
  }

  for (int i = 0; i < 1024; ++i) {
    c.Append(from);
  }
  ABSL_RAW_LOG(INFO, "Made a Cord with %zu bytes!", c.size());
  // Note: on a 32-bit build, this comes out to   2,818,048,000 bytes.
  // Note: on a 64-bit build, this comes out to 171,932,385,280 bytes.
}

static abslx::Cord MakeExternalCord(int size) {
  char* buffer = new char[size];
  memset(buffer, 'x', size);
  abslx::Cord cord;
  cord.Append(abslx::MakeCordFromExternal(
      abslx::string_view(buffer, size),
      [](abslx::string_view s) { delete[] s.data(); }));
  return cord;
}

// Extern to fool clang that this is not constant. Needed to suppress
// a warning of unsafe code we want to test.
extern bool my_unique_true_boolean;
bool my_unique_true_boolean = true;

TEST(Cord, Assignment) {
  abslx::Cord x(abslx::string_view("hi there"));
  abslx::Cord y(x);
  ASSERT_EQ(std::string(x), "hi there");
  ASSERT_EQ(std::string(y), "hi there");
  ASSERT_TRUE(x == y);
  ASSERT_TRUE(x <= y);
  ASSERT_TRUE(y <= x);

  x = abslx::string_view("foo");
  ASSERT_EQ(std::string(x), "foo");
  ASSERT_EQ(std::string(y), "hi there");
  ASSERT_TRUE(x < y);
  ASSERT_TRUE(y > x);
  ASSERT_TRUE(x != y);
  ASSERT_TRUE(x <= y);
  ASSERT_TRUE(y >= x);

  x = "foo";
  ASSERT_EQ(x, "foo");

  // Test that going from inline rep to tree we don't leak memory.
  std::vector<std::pair<abslx::string_view, abslx::string_view>>
      test_string_pairs = {{"hi there", "foo"},
                           {"loooooong coooooord", "short cord"},
                           {"short cord", "loooooong coooooord"},
                           {"loooooong coooooord1", "loooooong coooooord2"}};
  for (std::pair<abslx::string_view, abslx::string_view> test_strings :
       test_string_pairs) {
    abslx::Cord tmp(test_strings.first);
    abslx::Cord z(std::move(tmp));
    ASSERT_EQ(std::string(z), test_strings.first);
    tmp = test_strings.second;
    z = std::move(tmp);
    ASSERT_EQ(std::string(z), test_strings.second);
  }
  {
    // Test that self-move assignment doesn't crash/leak.
    // Do not write such code!
    abslx::Cord my_small_cord("foo");
    abslx::Cord my_big_cord("loooooong coooooord");
    // Bypass clang's warning on self move-assignment.
    abslx::Cord* my_small_alias =
        my_unique_true_boolean ? &my_small_cord : &my_big_cord;
    abslx::Cord* my_big_alias =
        !my_unique_true_boolean ? &my_small_cord : &my_big_cord;

    *my_small_alias = std::move(my_small_cord);
    *my_big_alias = std::move(my_big_cord);
    // my_small_cord and my_big_cord are in an unspecified but valid
    // state, and will be correctly destroyed here.
  }
}

TEST(Cord, StartsEndsWith) {
  abslx::Cord x(abslx::string_view("abcde"));
  abslx::Cord empty("");

  ASSERT_TRUE(x.StartsWith(abslx::Cord("abcde")));
  ASSERT_TRUE(x.StartsWith(abslx::Cord("abc")));
  ASSERT_TRUE(x.StartsWith(abslx::Cord("")));
  ASSERT_TRUE(empty.StartsWith(abslx::Cord("")));
  ASSERT_TRUE(x.EndsWith(abslx::Cord("abcde")));
  ASSERT_TRUE(x.EndsWith(abslx::Cord("cde")));
  ASSERT_TRUE(x.EndsWith(abslx::Cord("")));
  ASSERT_TRUE(empty.EndsWith(abslx::Cord("")));

  ASSERT_TRUE(!x.StartsWith(abslx::Cord("xyz")));
  ASSERT_TRUE(!empty.StartsWith(abslx::Cord("xyz")));
  ASSERT_TRUE(!x.EndsWith(abslx::Cord("xyz")));
  ASSERT_TRUE(!empty.EndsWith(abslx::Cord("xyz")));

  ASSERT_TRUE(x.StartsWith("abcde"));
  ASSERT_TRUE(x.StartsWith("abc"));
  ASSERT_TRUE(x.StartsWith(""));
  ASSERT_TRUE(empty.StartsWith(""));
  ASSERT_TRUE(x.EndsWith("abcde"));
  ASSERT_TRUE(x.EndsWith("cde"));
  ASSERT_TRUE(x.EndsWith(""));
  ASSERT_TRUE(empty.EndsWith(""));

  ASSERT_TRUE(!x.StartsWith("xyz"));
  ASSERT_TRUE(!empty.StartsWith("xyz"));
  ASSERT_TRUE(!x.EndsWith("xyz"));
  ASSERT_TRUE(!empty.EndsWith("xyz"));
}

TEST(Cord, Subcord) {
  RandomEngine rng(testing::GTEST_FLAG(random_seed));
  const std::string s = RandomLowercaseString(&rng, 1024);

  abslx::Cord a;
  AppendWithFragments(s, &rng, &a);
  ASSERT_EQ(s.size(), a.size());

  // Check subcords of a, from a variety of interesting points.
  std::set<size_t> positions;
  for (int i = 0; i <= 32; ++i) {
    positions.insert(i);
    positions.insert(i * 32 - 1);
    positions.insert(i * 32);
    positions.insert(i * 32 + 1);
    positions.insert(a.size() - i);
  }
  positions.insert(237);
  positions.insert(732);
  for (size_t pos : positions) {
    if (pos > a.size()) continue;
    for (size_t end_pos : positions) {
      if (end_pos < pos || end_pos > a.size()) continue;
      abslx::Cord sa = a.Subcord(pos, end_pos - pos);
      ASSERT_EQ(abslx::string_view(s).substr(pos, end_pos - pos),
                std::string(sa))
          << a;
    }
  }

  // Do the same thing for an inline cord.
  const std::string sh = "short";
  abslx::Cord c(sh);
  for (size_t pos = 0; pos <= sh.size(); ++pos) {
    for (size_t n = 0; n <= sh.size() - pos; ++n) {
      abslx::Cord sc = c.Subcord(pos, n);
      ASSERT_EQ(sh.substr(pos, n), std::string(sc)) << c;
    }
  }

  // Check subcords of subcords.
  abslx::Cord sa = a.Subcord(0, a.size());
  std::string ss = s.substr(0, s.size());
  while (sa.size() > 1) {
    sa = sa.Subcord(1, sa.size() - 2);
    ss = ss.substr(1, ss.size() - 2);
    ASSERT_EQ(ss, std::string(sa)) << a;
    if (HasFailure()) break;  // halt cascade
  }

  // It is OK to ask for too much.
  sa = a.Subcord(0, a.size() + 1);
  EXPECT_EQ(s, std::string(sa));

  // It is OK to ask for something beyond the end.
  sa = a.Subcord(a.size() + 1, 0);
  EXPECT_TRUE(sa.empty());
  sa = a.Subcord(a.size() + 1, 1);
  EXPECT_TRUE(sa.empty());
}

TEST(Cord, Swap) {
  abslx::string_view a("Dexter");
  abslx::string_view b("Mandark");
  abslx::Cord x(a);
  abslx::Cord y(b);
  swap(x, y);
  ASSERT_EQ(x, abslx::Cord(b));
  ASSERT_EQ(y, abslx::Cord(a));
  x.swap(y);
  ASSERT_EQ(x, abslx::Cord(a));
  ASSERT_EQ(y, abslx::Cord(b));
}

static void VerifyCopyToString(const abslx::Cord& cord) {
  std::string initially_empty;
  abslx::CopyCordToString(cord, &initially_empty);
  EXPECT_EQ(initially_empty, cord);

  constexpr size_t kInitialLength = 1024;
  std::string has_initial_contents(kInitialLength, 'x');
  const char* address_before_copy = has_initial_contents.data();
  abslx::CopyCordToString(cord, &has_initial_contents);
  EXPECT_EQ(has_initial_contents, cord);

  if (cord.size() <= kInitialLength) {
    EXPECT_EQ(has_initial_contents.data(), address_before_copy)
        << "CopyCordToString allocated new string storage; "
           "has_initial_contents = \""
        << has_initial_contents << "\"";
  }
}

TEST(Cord, CopyToString) {
  VerifyCopyToString(abslx::Cord());
  VerifyCopyToString(abslx::Cord("small cord"));
  VerifyCopyToString(
      abslx::MakeFragmentedCord({"fragmented ", "cord ", "to ", "test ",
                                "copying ", "to ", "a ", "string."}));
}

TEST(TryFlat, Empty) {
  abslx::Cord c;
  EXPECT_EQ(c.TryFlat(), "");
}

TEST(TryFlat, Flat) {
  abslx::Cord c("hello");
  EXPECT_EQ(c.TryFlat(), "hello");
}

TEST(TryFlat, SubstrInlined) {
  abslx::Cord c("hello");
  c.RemovePrefix(1);
  EXPECT_EQ(c.TryFlat(), "ello");
}

TEST(TryFlat, SubstrFlat) {
  abslx::Cord c("longer than 15 bytes");
  abslx::Cord sub = abslx::CordTestPeer::MakeSubstring(c, 1, c.size() - 1);
  EXPECT_EQ(sub.TryFlat(), "onger than 15 bytes");
}

TEST(TryFlat, Concat) {
  abslx::Cord c = abslx::MakeFragmentedCord({"hel", "lo"});
  EXPECT_EQ(c.TryFlat(), abslx::nullopt);
}

TEST(TryFlat, External) {
  abslx::Cord c = abslx::MakeCordFromExternal("hell", [](abslx::string_view) {});
  EXPECT_EQ(c.TryFlat(), "hell");
}

TEST(TryFlat, SubstrExternal) {
  abslx::Cord c = abslx::MakeCordFromExternal("hell", [](abslx::string_view) {});
  abslx::Cord sub = abslx::CordTestPeer::MakeSubstring(c, 1, c.size() - 1);
  EXPECT_EQ(sub.TryFlat(), "ell");
}

TEST(TryFlat, SubstrConcat) {
  abslx::Cord c = abslx::MakeFragmentedCord({"hello", " world"});
  abslx::Cord sub = abslx::CordTestPeer::MakeSubstring(c, 1, c.size() - 1);
  EXPECT_EQ(sub.TryFlat(), abslx::nullopt);
  c.RemovePrefix(1);
  EXPECT_EQ(c.TryFlat(), abslx::nullopt);
}

TEST(TryFlat, CommonlyAssumedInvariants) {
  // The behavior tested below is not part of the API contract of Cord, but it's
  // something we intend to be true in our current implementation.  This test
  // exists to detect and prevent accidental breakage of the implementation.
  abslx::string_view fragments[] = {"A fragmented test",
                                   " cord",
                                   " to test subcords",
                                   " of ",
                                   "a",
                                   " cord for",
                                   " each chunk "
                                   "returned by the ",
                                   "iterator"};
  abslx::Cord c = abslx::MakeFragmentedCord(fragments);
  int fragment = 0;
  int offset = 0;
  abslx::Cord::CharIterator itc = c.char_begin();
  for (abslx::string_view sv : c.Chunks()) {
    abslx::string_view expected = fragments[fragment];
    abslx::Cord subcord1 = c.Subcord(offset, sv.length());
    abslx::Cord subcord2 = abslx::Cord::AdvanceAndRead(&itc, sv.size());
    EXPECT_EQ(subcord1.TryFlat(), expected);
    EXPECT_EQ(subcord2.TryFlat(), expected);
    ++fragment;
    offset += sv.length();
  }
}

static bool IsFlat(const abslx::Cord& c) {
  return c.chunk_begin() == c.chunk_end() || ++c.chunk_begin() == c.chunk_end();
}

static void VerifyFlatten(abslx::Cord c) {
  std::string old_contents(c);
  abslx::string_view old_flat;
  bool already_flat_and_non_empty = IsFlat(c) && !c.empty();
  if (already_flat_and_non_empty) {
    old_flat = *c.chunk_begin();
  }
  abslx::string_view new_flat = c.Flatten();

  // Verify that the contents of the flattened Cord are correct.
  EXPECT_EQ(new_flat, old_contents);
  EXPECT_EQ(std::string(c), old_contents);

  // If the Cord contained data and was already flat, verify that the data
  // wasn't copied.
  if (already_flat_and_non_empty) {
    EXPECT_EQ(old_flat.data(), new_flat.data())
        << "Allocated new memory even though the Cord was already flat.";
  }

  // Verify that the flattened Cord is in fact flat.
  EXPECT_TRUE(IsFlat(c));
}

TEST(Cord, Flatten) {
  VerifyFlatten(abslx::Cord());
  VerifyFlatten(abslx::Cord("small cord"));
  VerifyFlatten(abslx::Cord("larger than small buffer optimization"));
  VerifyFlatten(abslx::MakeFragmentedCord({"small ", "fragmented ", "cord"}));

  // Test with a cord that is longer than the largest flat buffer
  RandomEngine rng(testing::GTEST_FLAG(random_seed));
  VerifyFlatten(abslx::Cord(RandomLowercaseString(&rng, 8192)));
}

// Test data
namespace {
class TestData {
 private:
  std::vector<std::string> data_;

  // Return a std::string of the specified length.
  static std::string MakeString(int length) {
    std::string result;
    char buf[30];
    snprintf(buf, sizeof(buf), "(%d)", length);
    while (result.size() < length) {
      result += buf;
    }
    result.resize(length);
    return result;
  }

 public:
  TestData() {
    // short strings increasing in length by one
    for (int i = 0; i < 30; i++) {
      data_.push_back(MakeString(i));
    }

    // strings around half kMaxFlatLength
    static const int kMaxFlatLength = 4096 - 9;
    static const int kHalf = kMaxFlatLength / 2;

    for (int i = -10; i <= +10; i++) {
      data_.push_back(MakeString(kHalf + i));
    }

    for (int i = -10; i <= +10; i++) {
      data_.push_back(MakeString(kMaxFlatLength + i));
    }
  }

  size_t size() const { return data_.size(); }
  const std::string& data(size_t i) const { return data_[i]; }
};
}  // namespace

TEST(Cord, MultipleLengths) {
  TestData d;
  for (size_t i = 0; i < d.size(); i++) {
    std::string a = d.data(i);

    {  // Construct from Cord
      abslx::Cord tmp(a);
      abslx::Cord x(tmp);
      EXPECT_EQ(a, std::string(x)) << "'" << a << "'";
    }

    {  // Construct from abslx::string_view
      abslx::Cord x(a);
      EXPECT_EQ(a, std::string(x)) << "'" << a << "'";
    }

    {  // Append cord to self
      abslx::Cord self(a);
      self.Append(self);
      EXPECT_EQ(a + a, std::string(self)) << "'" << a << "' + '" << a << "'";
    }

    {  // Prepend cord to self
      abslx::Cord self(a);
      self.Prepend(self);
      EXPECT_EQ(a + a, std::string(self)) << "'" << a << "' + '" << a << "'";
    }

    // Try to append/prepend others
    for (size_t j = 0; j < d.size(); j++) {
      std::string b = d.data(j);

      {  // CopyFrom Cord
        abslx::Cord x(a);
        abslx::Cord y(b);
        x = y;
        EXPECT_EQ(b, std::string(x)) << "'" << a << "' + '" << b << "'";
      }

      {  // CopyFrom abslx::string_view
        abslx::Cord x(a);
        x = b;
        EXPECT_EQ(b, std::string(x)) << "'" << a << "' + '" << b << "'";
      }

      {  // Cord::Append(Cord)
        abslx::Cord x(a);
        abslx::Cord y(b);
        x.Append(y);
        EXPECT_EQ(a + b, std::string(x)) << "'" << a << "' + '" << b << "'";
      }

      {  // Cord::Append(abslx::string_view)
        abslx::Cord x(a);
        x.Append(b);
        EXPECT_EQ(a + b, std::string(x)) << "'" << a << "' + '" << b << "'";
      }

      {  // Cord::Prepend(Cord)
        abslx::Cord x(a);
        abslx::Cord y(b);
        x.Prepend(y);
        EXPECT_EQ(b + a, std::string(x)) << "'" << b << "' + '" << a << "'";
      }

      {  // Cord::Prepend(abslx::string_view)
        abslx::Cord x(a);
        x.Prepend(b);
        EXPECT_EQ(b + a, std::string(x)) << "'" << b << "' + '" << a << "'";
      }
    }
  }
}

namespace {

TEST(Cord, RemoveSuffixWithExternalOrSubstring) {
  abslx::Cord cord = abslx::MakeCordFromExternal(
      "foo bar baz", [](abslx::string_view s) { DoNothing(s, nullptr); });

  EXPECT_EQ("foo bar baz", std::string(cord));

  // This RemoveSuffix() will wrap the EXTERNAL node in a SUBSTRING node.
  cord.RemoveSuffix(4);
  EXPECT_EQ("foo bar", std::string(cord));

  // This RemoveSuffix() will adjust the SUBSTRING node in-place.
  cord.RemoveSuffix(4);
  EXPECT_EQ("foo", std::string(cord));
}

TEST(Cord, RemoveSuffixMakesZeroLengthNode) {
  abslx::Cord c;
  c.Append(abslx::Cord(std::string(100, 'x')));
  abslx::Cord other_ref = c;  // Prevent inplace appends
  c.Append(abslx::Cord(std::string(200, 'y')));
  c.RemoveSuffix(200);
  EXPECT_EQ(std::string(100, 'x'), std::string(c));
}

}  // namespace

// CordSpliceTest contributed by hendrie.
namespace {

// Create a cord with an external memory block filled with 'z'
abslx::Cord CordWithZedBlock(size_t size) {
  char* data = new char[size];
  if (size > 0) {
    memset(data, 'z', size);
  }
  abslx::Cord cord = abslx::MakeCordFromExternal(
      abslx::string_view(data, size),
      [](abslx::string_view s) { delete[] s.data(); });
  return cord;
}

// Establish that ZedBlock does what we think it does.
TEST(CordSpliceTest, ZedBlock) {
  abslx::Cord blob = CordWithZedBlock(10);
  EXPECT_EQ(10, blob.size());
  std::string s;
  abslx::CopyCordToString(blob, &s);
  EXPECT_EQ("zzzzzzzzzz", s);
}

TEST(CordSpliceTest, ZedBlock0) {
  abslx::Cord blob = CordWithZedBlock(0);
  EXPECT_EQ(0, blob.size());
  std::string s;
  abslx::CopyCordToString(blob, &s);
  EXPECT_EQ("", s);
}

TEST(CordSpliceTest, ZedBlockSuffix1) {
  abslx::Cord blob = CordWithZedBlock(10);
  EXPECT_EQ(10, blob.size());
  abslx::Cord suffix(blob);
  suffix.RemovePrefix(9);
  EXPECT_EQ(1, suffix.size());
  std::string s;
  abslx::CopyCordToString(suffix, &s);
  EXPECT_EQ("z", s);
}

// Remove all of a prefix block
TEST(CordSpliceTest, ZedBlockSuffix0) {
  abslx::Cord blob = CordWithZedBlock(10);
  EXPECT_EQ(10, blob.size());
  abslx::Cord suffix(blob);
  suffix.RemovePrefix(10);
  EXPECT_EQ(0, suffix.size());
  std::string s;
  abslx::CopyCordToString(suffix, &s);
  EXPECT_EQ("", s);
}

abslx::Cord BigCord(size_t len, char v) {
  std::string s(len, v);
  return abslx::Cord(s);
}

// Splice block into cord.
abslx::Cord SpliceCord(const abslx::Cord& blob, int64_t offset,
                      const abslx::Cord& block) {
  ABSL_RAW_CHECK(offset >= 0, "");
  ABSL_RAW_CHECK(offset + block.size() <= blob.size(), "");
  abslx::Cord result(blob);
  result.RemoveSuffix(blob.size() - offset);
  result.Append(block);
  abslx::Cord suffix(blob);
  suffix.RemovePrefix(offset + block.size());
  result.Append(suffix);
  ABSL_RAW_CHECK(blob.size() == result.size(), "");
  return result;
}

// Taking an empty suffix of a block breaks appending.
TEST(CordSpliceTest, RemoveEntireBlock1) {
  abslx::Cord zero = CordWithZedBlock(10);
  abslx::Cord suffix(zero);
  suffix.RemovePrefix(10);
  abslx::Cord result;
  result.Append(suffix);
}

TEST(CordSpliceTest, RemoveEntireBlock2) {
  abslx::Cord zero = CordWithZedBlock(10);
  abslx::Cord prefix(zero);
  prefix.RemoveSuffix(10);
  abslx::Cord suffix(zero);
  suffix.RemovePrefix(10);
  abslx::Cord result(prefix);
  result.Append(suffix);
}

TEST(CordSpliceTest, RemoveEntireBlock3) {
  abslx::Cord blob = CordWithZedBlock(10);
  abslx::Cord block = BigCord(10, 'b');
  blob = SpliceCord(blob, 0, block);
}

struct CordCompareTestCase {
  template <typename LHS, typename RHS>
  CordCompareTestCase(const LHS& lhs, const RHS& rhs)
      : lhs_cord(lhs), rhs_cord(rhs) {}

  abslx::Cord lhs_cord;
  abslx::Cord rhs_cord;
};

const auto sign = [](int x) { return x == 0 ? 0 : (x > 0 ? 1 : -1); };

void VerifyComparison(const CordCompareTestCase& test_case) {
  std::string lhs_string(test_case.lhs_cord);
  std::string rhs_string(test_case.rhs_cord);
  int expected = sign(lhs_string.compare(rhs_string));
  EXPECT_EQ(expected, test_case.lhs_cord.Compare(test_case.rhs_cord))
      << "LHS=" << lhs_string << "; RHS=" << rhs_string;
  EXPECT_EQ(expected, test_case.lhs_cord.Compare(rhs_string))
      << "LHS=" << lhs_string << "; RHS=" << rhs_string;
  EXPECT_EQ(-expected, test_case.rhs_cord.Compare(test_case.lhs_cord))
      << "LHS=" << rhs_string << "; RHS=" << lhs_string;
  EXPECT_EQ(-expected, test_case.rhs_cord.Compare(lhs_string))
      << "LHS=" << rhs_string << "; RHS=" << lhs_string;
}

TEST(Cord, Compare) {
  abslx::Cord subcord("aaaaaBBBBBcccccDDDDD");
  subcord = subcord.Subcord(3, 10);

  abslx::Cord tmp("aaaaaaaaaaaaaaaa");
  tmp.Append("BBBBBBBBBBBBBBBB");
  abslx::Cord concat = abslx::Cord("cccccccccccccccc");
  concat.Append("DDDDDDDDDDDDDDDD");
  concat.Prepend(tmp);

  abslx::Cord concat2("aaaaaaaaaaaaa");
  concat2.Append("aaaBBBBBBBBBBBBBBBBccccc");
  concat2.Append("cccccccccccDDDDDDDDDDDDDD");
  concat2.Append("DD");

  std::vector<CordCompareTestCase> test_cases = {{
      // Inline cords
      {"abcdef", "abcdef"},
      {"abcdef", "abcdee"},
      {"abcdef", "abcdeg"},
      {"bbcdef", "abcdef"},
      {"bbcdef", "abcdeg"},
      {"abcdefa", "abcdef"},
      {"abcdef", "abcdefa"},

      // Small flat cords
      {"aaaaaBBBBBcccccDDDDD", "aaaaaBBBBBcccccDDDDD"},
      {"aaaaaBBBBBcccccDDDDD", "aaaaaBBBBBxccccDDDDD"},
      {"aaaaaBBBBBcxcccDDDDD", "aaaaaBBBBBcccccDDDDD"},
      {"aaaaaBBBBBxccccDDDDD", "aaaaaBBBBBcccccDDDDX"},
      {"aaaaaBBBBBcccccDDDDDa", "aaaaaBBBBBcccccDDDDD"},
      {"aaaaaBBBBBcccccDDDDD", "aaaaaBBBBBcccccDDDDDa"},

      // Subcords
      {subcord, subcord},
      {subcord, "aaBBBBBccc"},
      {subcord, "aaBBBBBccd"},
      {subcord, "aaBBBBBccb"},
      {subcord, "aaBBBBBxcb"},
      {subcord, "aaBBBBBccca"},
      {subcord, "aaBBBBBcc"},

      // Concats
      {concat, concat},
      {concat,
       "aaaaaaaaaaaaaaaaBBBBBBBBBBBBBBBBccccccccccccccccDDDDDDDDDDDDDDDD"},
      {concat,
       "aaaaaaaaaaaaaaaaBBBBBBBBBBBBBBBBcccccccccccccccxDDDDDDDDDDDDDDDD"},
      {concat,
       "aaaaaaaaaaaaaaaaBBBBBBBBBBBBBBBBacccccccccccccccDDDDDDDDDDDDDDDD"},
      {concat,
       "aaaaaaaaaaaaaaaaBBBBBBBBBBBBBBBBccccccccccccccccDDDDDDDDDDDDDDD"},
      {concat,
       "aaaaaaaaaaaaaaaaBBBBBBBBBBBBBBBBccccccccccccccccDDDDDDDDDDDDDDDDe"},

      {concat, concat2},
  }};

  for (const auto& tc : test_cases) {
    VerifyComparison(tc);
  }
}

TEST(Cord, CompareAfterAssign) {
  abslx::Cord a("aaaaaa1111111");
  abslx::Cord b("aaaaaa2222222");
  a = "cccccc";
  b = "cccccc";
  EXPECT_EQ(a, b);
  EXPECT_FALSE(a < b);

  a = "aaaa";
  b = "bbbbb";
  a = "";
  b = "";
  EXPECT_EQ(a, b);
  EXPECT_FALSE(a < b);
}

// Test CompareTo() and ComparePrefix() against string and substring
// comparison methods from basic_string.
static void TestCompare(const abslx::Cord& c, const abslx::Cord& d,
                        RandomEngine* rng) {
  typedef std::basic_string<uint8_t> ustring;
  ustring cs(reinterpret_cast<const uint8_t*>(std::string(c).data()), c.size());
  ustring ds(reinterpret_cast<const uint8_t*>(std::string(d).data()), d.size());
  // ustring comparison is ideal because we expect Cord comparisons to be
  // based on unsigned byte comparisons regardless of whether char is signed.
  int expected = sign(cs.compare(ds));
  EXPECT_EQ(expected, sign(c.Compare(d))) << c << ", " << d;
}

TEST(Compare, ComparisonIsUnsigned) {
  RandomEngine rng(testing::GTEST_FLAG(random_seed));
  std::uniform_int_distribution<uint32_t> uniform_uint8(0, 255);
  char x = static_cast<char>(uniform_uint8(rng));
  TestCompare(
      abslx::Cord(std::string(GetUniformRandomUpTo(&rng, 100), x)),
      abslx::Cord(std::string(GetUniformRandomUpTo(&rng, 100), x ^ 0x80)), &rng);
}

TEST(Compare, RandomComparisons) {
  const int kIters = 5000;
  RandomEngine rng(testing::GTEST_FLAG(random_seed));

  int n = GetUniformRandomUpTo(&rng, 5000);
  abslx::Cord a[] = {MakeExternalCord(n),
                    abslx::Cord("ant"),
                    abslx::Cord("elephant"),
                    abslx::Cord("giraffe"),
                    abslx::Cord(std::string(GetUniformRandomUpTo(&rng, 100),
                                           GetUniformRandomUpTo(&rng, 100))),
                    abslx::Cord(""),
                    abslx::Cord("x"),
                    abslx::Cord("A"),
                    abslx::Cord("B"),
                    abslx::Cord("C")};
  for (int i = 0; i < kIters; i++) {
    abslx::Cord c, d;
    for (int j = 0; j < (i % 7) + 1; j++) {
      c.Append(a[GetUniformRandomUpTo(&rng, ABSL_ARRAYSIZE(a))]);
      d.Append(a[GetUniformRandomUpTo(&rng, ABSL_ARRAYSIZE(a))]);
    }
    std::bernoulli_distribution coin_flip(0.5);
    TestCompare(coin_flip(rng) ? c : abslx::Cord(std::string(c)),
                coin_flip(rng) ? d : abslx::Cord(std::string(d)), &rng);
  }
}

template <typename T1, typename T2>
void CompareOperators() {
  const T1 a("a");
  const T2 b("b");

  EXPECT_TRUE(a == a);
  // For pointer type (i.e. `const char*`), operator== compares the address
  // instead of the string, so `a == const char*("a")` isn't necessarily true.
  EXPECT_TRUE(std::is_pointer<T1>::value || a == T1("a"));
  EXPECT_TRUE(std::is_pointer<T2>::value || a == T2("a"));
  EXPECT_FALSE(a == b);

  EXPECT_TRUE(a != b);
  EXPECT_FALSE(a != a);

  EXPECT_TRUE(a < b);
  EXPECT_FALSE(b < a);

  EXPECT_TRUE(b > a);
  EXPECT_FALSE(a > b);

  EXPECT_TRUE(a >= a);
  EXPECT_TRUE(b >= a);
  EXPECT_FALSE(a >= b);

  EXPECT_TRUE(a <= a);
  EXPECT_TRUE(a <= b);
  EXPECT_FALSE(b <= a);
}

TEST(ComparisonOperators, Cord_Cord) {
  CompareOperators<abslx::Cord, abslx::Cord>();
}

TEST(ComparisonOperators, Cord_StringPiece) {
  CompareOperators<abslx::Cord, abslx::string_view>();
}

TEST(ComparisonOperators, StringPiece_Cord) {
  CompareOperators<abslx::string_view, abslx::Cord>();
}

TEST(ComparisonOperators, Cord_string) {
  CompareOperators<abslx::Cord, std::string>();
}

TEST(ComparisonOperators, string_Cord) {
  CompareOperators<std::string, abslx::Cord>();
}

TEST(ComparisonOperators, stdstring_Cord) {
  CompareOperators<std::string, abslx::Cord>();
}

TEST(ComparisonOperators, Cord_stdstring) {
  CompareOperators<abslx::Cord, std::string>();
}

TEST(ComparisonOperators, charstar_Cord) {
  CompareOperators<const char*, abslx::Cord>();
}

TEST(ComparisonOperators, Cord_charstar) {
  CompareOperators<abslx::Cord, const char*>();
}

TEST(ConstructFromExternal, ReleaserInvoked) {
  // Empty external memory means the releaser should be called immediately.
  {
    bool invoked = false;
    auto releaser = [&invoked](abslx::string_view) { invoked = true; };
    {
      auto c = abslx::MakeCordFromExternal("", releaser);
      EXPECT_TRUE(invoked);
    }
  }

  // If the size of the data is small enough, a future constructor
  // implementation may copy the bytes and immediately invoke the releaser
  // instead of creating an external node. We make a large dummy std::string to
  // make this test independent of such an optimization.
  std::string large_dummy(2048, 'c');
  {
    bool invoked = false;
    auto releaser = [&invoked](abslx::string_view) { invoked = true; };
    {
      auto c = abslx::MakeCordFromExternal(large_dummy, releaser);
      EXPECT_FALSE(invoked);
    }
    EXPECT_TRUE(invoked);
  }

  {
    bool invoked = false;
    auto releaser = [&invoked](abslx::string_view) { invoked = true; };
    {
      abslx::Cord copy;
      {
        auto c = abslx::MakeCordFromExternal(large_dummy, releaser);
        copy = c;
        EXPECT_FALSE(invoked);
      }
      EXPECT_FALSE(invoked);
    }
    EXPECT_TRUE(invoked);
  }
}

TEST(ConstructFromExternal, CompareContents) {
  RandomEngine rng(testing::GTEST_FLAG(random_seed));

  for (int length = 1; length <= 2048; length *= 2) {
    std::string data = RandomLowercaseString(&rng, length);
    auto* external = new std::string(data);
    auto cord =
        abslx::MakeCordFromExternal(*external, [external](abslx::string_view sv) {
          EXPECT_EQ(external->data(), sv.data());
          EXPECT_EQ(external->size(), sv.size());
          delete external;
        });
    EXPECT_EQ(data, cord);
  }
}

TEST(ConstructFromExternal, LargeReleaser) {
  RandomEngine rng(testing::GTEST_FLAG(random_seed));
  constexpr size_t kLength = 256;
  std::string data = RandomLowercaseString(&rng, kLength);
  std::array<char, kLength> data_array;
  for (size_t i = 0; i < kLength; ++i) data_array[i] = data[i];
  bool invoked = false;
  auto releaser = [data_array, &invoked](abslx::string_view data) {
    EXPECT_EQ(data, abslx::string_view(data_array.data(), data_array.size()));
    invoked = true;
  };
  (void)abslx::MakeCordFromExternal(data, releaser);
  EXPECT_TRUE(invoked);
}

TEST(ConstructFromExternal, FunctionPointerReleaser) {
  static abslx::string_view data("hello world");
  static bool invoked;
  auto* releaser =
      static_cast<void (*)(abslx::string_view)>([](abslx::string_view sv) {
        EXPECT_EQ(data, sv);
        invoked = true;
      });
  invoked = false;
  (void)abslx::MakeCordFromExternal(data, releaser);
  EXPECT_TRUE(invoked);

  invoked = false;
  (void)abslx::MakeCordFromExternal(data, *releaser);
  EXPECT_TRUE(invoked);
}

TEST(ConstructFromExternal, MoveOnlyReleaser) {
  struct Releaser {
    explicit Releaser(bool* invoked) : invoked(invoked) {}
    Releaser(Releaser&& other) noexcept : invoked(other.invoked) {}
    void operator()(abslx::string_view) const { *invoked = true; }

    bool* invoked;
  };

  bool invoked = false;
  (void)abslx::MakeCordFromExternal("dummy", Releaser(&invoked));
  EXPECT_TRUE(invoked);
}

TEST(ConstructFromExternal, NoArgLambda) {
  bool invoked = false;
  (void)abslx::MakeCordFromExternal("dummy", [&invoked]() { invoked = true; });
  EXPECT_TRUE(invoked);
}

TEST(ConstructFromExternal, StringViewArgLambda) {
  bool invoked = false;
  (void)abslx::MakeCordFromExternal(
      "dummy", [&invoked](abslx::string_view) { invoked = true; });
  EXPECT_TRUE(invoked);
}

TEST(ConstructFromExternal, NonTrivialReleaserDestructor) {
  struct Releaser {
    explicit Releaser(bool* destroyed) : destroyed(destroyed) {}
    ~Releaser() { *destroyed = true; }
    void operator()(abslx::string_view) const {}

    bool* destroyed;
  };

  bool destroyed = false;
  Releaser releaser(&destroyed);
  (void)abslx::MakeCordFromExternal("dummy", releaser);
  EXPECT_TRUE(destroyed);
}

TEST(ConstructFromExternal, ReferenceQualifierOverloads) {
  struct Releaser {
    void operator()(abslx::string_view) & { *lvalue_invoked = true; }
    void operator()(abslx::string_view) && { *rvalue_invoked = true; }

    bool* lvalue_invoked;
    bool* rvalue_invoked;
  };

  bool lvalue_invoked = false;
  bool rvalue_invoked = false;
  Releaser releaser = {&lvalue_invoked, &rvalue_invoked};
  (void)abslx::MakeCordFromExternal("", releaser);
  EXPECT_FALSE(lvalue_invoked);
  EXPECT_TRUE(rvalue_invoked);
  rvalue_invoked = false;

  (void)abslx::MakeCordFromExternal("dummy", releaser);
  EXPECT_FALSE(lvalue_invoked);
  EXPECT_TRUE(rvalue_invoked);
  rvalue_invoked = false;

  // NOLINTNEXTLINE: suppress clang-tidy std::move on trivially copyable type.
  (void)abslx::MakeCordFromExternal("dummy", std::move(releaser));
  EXPECT_FALSE(lvalue_invoked);
  EXPECT_TRUE(rvalue_invoked);
}

TEST(ExternalMemory, BasicUsage) {
  static const char* strings[] = {"", "hello", "there"};
  for (const char* str : strings) {
    abslx::Cord dst("(prefix)");
    AddExternalMemory(str, &dst);
    dst.Append("(suffix)");
    EXPECT_EQ((std::string("(prefix)") + str + std::string("(suffix)")),
              std::string(dst));
  }
}

TEST(ExternalMemory, RemovePrefixSuffix) {
  // Exhaustively try all sub-strings.
  abslx::Cord cord = MakeComposite();
  std::string s = std::string(cord);
  for (int offset = 0; offset <= s.size(); offset++) {
    for (int length = 0; length <= s.size() - offset; length++) {
      abslx::Cord result(cord);
      result.RemovePrefix(offset);
      result.RemoveSuffix(result.size() - length);
      EXPECT_EQ(s.substr(offset, length), std::string(result))
          << offset << " " << length;
    }
  }
}

TEST(ExternalMemory, Get) {
  abslx::Cord cord("hello");
  AddExternalMemory(" world!", &cord);
  AddExternalMemory(" how are ", &cord);
  cord.Append(" you?");
  std::string s = std::string(cord);
  for (int i = 0; i < s.size(); i++) {
    EXPECT_EQ(s[i], cord[i]);
  }
}

// CordMemoryUsage tests verify the correctness of the EstimatedMemoryUsage()
// These tests take into account that the reported memory usage is approximate
// and non-deterministic. For all tests, We verify that the reported memory
// usage is larger than `size()`, and less than `size() * 1.5` as a cord should
// never reserve more 'extra' capacity than half of its size as it grows.
// Additionally we have some whiteboxed expectations based on our knowledge of
// the layout and size of empty and inlined cords, and flat nodes.

TEST(CordMemoryUsage, Empty) {
  EXPECT_EQ(sizeof(abslx::Cord), abslx::Cord().EstimatedMemoryUsage());
}

TEST(CordMemoryUsage, Embedded) {
  abslx::Cord a("hello");
  EXPECT_EQ(a.EstimatedMemoryUsage(), sizeof(abslx::Cord));
}

TEST(CordMemoryUsage, EmbeddedAppend) {
  abslx::Cord a("a");
  abslx::Cord b("bcd");
  EXPECT_EQ(b.EstimatedMemoryUsage(), sizeof(abslx::Cord));
  a.Append(b);
  EXPECT_EQ(a.EstimatedMemoryUsage(), sizeof(abslx::Cord));
}

TEST(CordMemoryUsage, ExternalMemory) {
  static const int kLength = 1000;
  abslx::Cord cord;
  AddExternalMemory(std::string(kLength, 'x'), &cord);
  EXPECT_GT(cord.EstimatedMemoryUsage(), kLength);
  EXPECT_LE(cord.EstimatedMemoryUsage(), kLength * 1.5);
}

TEST(CordMemoryUsage, Flat) {
  static const int kLength = 125;
  abslx::Cord a(std::string(kLength, 'a'));
  EXPECT_GT(a.EstimatedMemoryUsage(), kLength);
  EXPECT_LE(a.EstimatedMemoryUsage(), kLength * 1.5);
}

TEST(CordMemoryUsage, AppendFlat) {
  using abslx::strings_internal::CordTestAccess;
  abslx::Cord a(std::string(CordTestAccess::MaxFlatLength(), 'a'));
  size_t length = a.EstimatedMemoryUsage();
  a.Append(std::string(CordTestAccess::MaxFlatLength(), 'b'));
  size_t delta = a.EstimatedMemoryUsage() - length;
  EXPECT_GT(delta, CordTestAccess::MaxFlatLength());
  EXPECT_LE(delta, CordTestAccess::MaxFlatLength() * 1.5);
}

// Regtest for a change that had to be rolled back because it expanded out
// of the InlineRep too soon, which was observable through MemoryUsage().
TEST(CordMemoryUsage, InlineRep) {
  constexpr size_t kMaxInline = 15;  // Cord::InlineRep::N
  const std::string small_string(kMaxInline, 'x');
  abslx::Cord c1(small_string);

  abslx::Cord c2;
  c2.Append(small_string);
  EXPECT_EQ(c1, c2);
  EXPECT_EQ(c1.EstimatedMemoryUsage(), c2.EstimatedMemoryUsage());
}

}  // namespace

// Regtest for 7510292 (fix a bug introduced by 7465150)
TEST(Cord, Concat_Append) {
  // Create a rep of type CONCAT
  abslx::Cord s1("foobarbarbarbarbar");
  s1.Append("abcdefgabcdefgabcdefgabcdefgabcdefgabcdefgabcdefg");
  size_t size = s1.size();

  // Create a copy of s1 and append to it.
  abslx::Cord s2 = s1;
  s2.Append("x");

  // 7465150 modifies s1 when it shouldn't.
  EXPECT_EQ(s1.size(), size);
  EXPECT_EQ(s2.size(), size + 1);
}

TEST(Cord, DiabolicalGrowth) {
  // This test exercises a diabolical Append(<one char>) on a cord, making the
  // cord shared before each Append call resulting in a terribly fragmented
  // resulting cord.
  // TODO(b/183983616): Apply some minimum compaction when copying a shared
  // source cord into a mutable copy for updates in CordRepRing.
  RandomEngine rng(testing::GTEST_FLAG(random_seed));
  const std::string expected = RandomLowercaseString(&rng, 5000);
  abslx::Cord cord;
  for (char c : expected) {
    abslx::Cord shared(cord);
    cord.Append(abslx::string_view(&c, 1));
  }
  std::string value;
  abslx::CopyCordToString(cord, &value);
  EXPECT_EQ(value, expected);
  ABSL_RAW_LOG(INFO, "Diabolical size allocated = %zu",
               cord.EstimatedMemoryUsage());
}

TEST(MakeFragmentedCord, MakeFragmentedCordFromInitializerList) {
  abslx::Cord fragmented =
      abslx::MakeFragmentedCord({"A ", "fragmented ", "Cord"});

  EXPECT_EQ("A fragmented Cord", fragmented);

  auto chunk_it = fragmented.chunk_begin();

  ASSERT_TRUE(chunk_it != fragmented.chunk_end());
  EXPECT_EQ("A ", *chunk_it);

  ASSERT_TRUE(++chunk_it != fragmented.chunk_end());
  EXPECT_EQ("fragmented ", *chunk_it);

  ASSERT_TRUE(++chunk_it != fragmented.chunk_end());
  EXPECT_EQ("Cord", *chunk_it);

  ASSERT_TRUE(++chunk_it == fragmented.chunk_end());
}

TEST(MakeFragmentedCord, MakeFragmentedCordFromVector) {
  std::vector<abslx::string_view> chunks = {"A ", "fragmented ", "Cord"};
  abslx::Cord fragmented = abslx::MakeFragmentedCord(chunks);

  EXPECT_EQ("A fragmented Cord", fragmented);

  auto chunk_it = fragmented.chunk_begin();

  ASSERT_TRUE(chunk_it != fragmented.chunk_end());
  EXPECT_EQ("A ", *chunk_it);

  ASSERT_TRUE(++chunk_it != fragmented.chunk_end());
  EXPECT_EQ("fragmented ", *chunk_it);

  ASSERT_TRUE(++chunk_it != fragmented.chunk_end());
  EXPECT_EQ("Cord", *chunk_it);

  ASSERT_TRUE(++chunk_it == fragmented.chunk_end());
}

TEST(CordChunkIterator, Traits) {
  static_assert(std::is_copy_constructible<abslx::Cord::ChunkIterator>::value,
                "");
  static_assert(std::is_copy_assignable<abslx::Cord::ChunkIterator>::value, "");

  // Move semantics to satisfy swappable via std::swap
  static_assert(std::is_move_constructible<abslx::Cord::ChunkIterator>::value,
                "");
  static_assert(std::is_move_assignable<abslx::Cord::ChunkIterator>::value, "");

  static_assert(
      std::is_same<
          std::iterator_traits<abslx::Cord::ChunkIterator>::iterator_category,
          std::input_iterator_tag>::value,
      "");
  static_assert(
      std::is_same<std::iterator_traits<abslx::Cord::ChunkIterator>::value_type,
                   abslx::string_view>::value,
      "");
  static_assert(
      std::is_same<
          std::iterator_traits<abslx::Cord::ChunkIterator>::difference_type,
          ptrdiff_t>::value,
      "");
  static_assert(
      std::is_same<std::iterator_traits<abslx::Cord::ChunkIterator>::pointer,
                   const abslx::string_view*>::value,
      "");
  static_assert(
      std::is_same<std::iterator_traits<abslx::Cord::ChunkIterator>::reference,
                   abslx::string_view>::value,
      "");
}

static void VerifyChunkIterator(const abslx::Cord& cord,
                                size_t expected_chunks) {
  EXPECT_EQ(cord.chunk_begin() == cord.chunk_end(), cord.empty()) << cord;
  EXPECT_EQ(cord.chunk_begin() != cord.chunk_end(), !cord.empty());

  abslx::Cord::ChunkRange range = cord.Chunks();
  EXPECT_EQ(range.begin() == range.end(), cord.empty());
  EXPECT_EQ(range.begin() != range.end(), !cord.empty());

  std::string content(cord);
  size_t pos = 0;
  auto pre_iter = cord.chunk_begin(), post_iter = cord.chunk_begin();
  size_t n_chunks = 0;
  while (pre_iter != cord.chunk_end() && post_iter != cord.chunk_end()) {
    EXPECT_FALSE(pre_iter == cord.chunk_end());   // NOLINT: explicitly test ==
    EXPECT_FALSE(post_iter == cord.chunk_end());  // NOLINT

    EXPECT_EQ(pre_iter, post_iter);
    EXPECT_EQ(*pre_iter, *post_iter);

    EXPECT_EQ(pre_iter->data(), (*pre_iter).data());
    EXPECT_EQ(pre_iter->size(), (*pre_iter).size());

    abslx::string_view chunk = *pre_iter;
    EXPECT_FALSE(chunk.empty());
    EXPECT_LE(pos + chunk.size(), content.size());
    EXPECT_EQ(abslx::string_view(content.c_str() + pos, chunk.size()), chunk);

    int n_equal_iterators = 0;
    for (abslx::Cord::ChunkIterator it = range.begin(); it != range.end();
         ++it) {
      n_equal_iterators += static_cast<int>(it == pre_iter);
    }
    EXPECT_EQ(n_equal_iterators, 1);

    ++pre_iter;
    EXPECT_EQ(*post_iter++, chunk);

    pos += chunk.size();
    ++n_chunks;
  }
  EXPECT_EQ(expected_chunks, n_chunks);
  EXPECT_EQ(pos, content.size());
  EXPECT_TRUE(pre_iter == cord.chunk_end());   // NOLINT: explicitly test ==
  EXPECT_TRUE(post_iter == cord.chunk_end());  // NOLINT
}

TEST(CordChunkIterator, Operations) {
  abslx::Cord empty_cord;
  VerifyChunkIterator(empty_cord, 0);

  abslx::Cord small_buffer_cord("small cord");
  VerifyChunkIterator(small_buffer_cord, 1);

  abslx::Cord flat_node_cord("larger than small buffer optimization");
  VerifyChunkIterator(flat_node_cord, 1);

  VerifyChunkIterator(
      abslx::MakeFragmentedCord({"a ", "small ", "fragmented ", "cord ", "for ",
                                "testing ", "chunk ", "iterations."}),
      8);

  abslx::Cord reused_nodes_cord(std::string(40, 'c'));
  reused_nodes_cord.Prepend(abslx::Cord(std::string(40, 'b')));
  reused_nodes_cord.Prepend(abslx::Cord(std::string(40, 'a')));
  size_t expected_chunks = 3;
  for (int i = 0; i < 8; ++i) {
    reused_nodes_cord.Prepend(reused_nodes_cord);
    expected_chunks *= 2;
    VerifyChunkIterator(reused_nodes_cord, expected_chunks);
  }

  RandomEngine rng(testing::GTEST_FLAG(random_seed));
  abslx::Cord flat_cord(RandomLowercaseString(&rng, 256));
  abslx::Cord subcords;
  for (int i = 0; i < 128; ++i) subcords.Prepend(flat_cord.Subcord(i, 128));
  VerifyChunkIterator(subcords, 128);
}

TEST(CordCharIterator, Traits) {
  static_assert(std::is_copy_constructible<abslx::Cord::CharIterator>::value,
                "");
  static_assert(std::is_copy_assignable<abslx::Cord::CharIterator>::value, "");

  // Move semantics to satisfy swappable via std::swap
  static_assert(std::is_move_constructible<abslx::Cord::CharIterator>::value,
                "");
  static_assert(std::is_move_assignable<abslx::Cord::CharIterator>::value, "");

  static_assert(
      std::is_same<
          std::iterator_traits<abslx::Cord::CharIterator>::iterator_category,
          std::input_iterator_tag>::value,
      "");
  static_assert(
      std::is_same<std::iterator_traits<abslx::Cord::CharIterator>::value_type,
                   char>::value,
      "");
  static_assert(
      std::is_same<
          std::iterator_traits<abslx::Cord::CharIterator>::difference_type,
          ptrdiff_t>::value,
      "");
  static_assert(
      std::is_same<std::iterator_traits<abslx::Cord::CharIterator>::pointer,
                   const char*>::value,
      "");
  static_assert(
      std::is_same<std::iterator_traits<abslx::Cord::CharIterator>::reference,
                   const char&>::value,
      "");
}

static void VerifyCharIterator(const abslx::Cord& cord) {
  EXPECT_EQ(cord.char_begin() == cord.char_end(), cord.empty());
  EXPECT_EQ(cord.char_begin() != cord.char_end(), !cord.empty());

  abslx::Cord::CharRange range = cord.Chars();
  EXPECT_EQ(range.begin() == range.end(), cord.empty());
  EXPECT_EQ(range.begin() != range.end(), !cord.empty());

  size_t i = 0;
  abslx::Cord::CharIterator pre_iter = cord.char_begin();
  abslx::Cord::CharIterator post_iter = cord.char_begin();
  std::string content(cord);
  while (pre_iter != cord.char_end() && post_iter != cord.char_end()) {
    EXPECT_FALSE(pre_iter == cord.char_end());   // NOLINT: explicitly test ==
    EXPECT_FALSE(post_iter == cord.char_end());  // NOLINT

    EXPECT_LT(i, cord.size());
    EXPECT_EQ(content[i], *pre_iter);

    EXPECT_EQ(pre_iter, post_iter);
    EXPECT_EQ(*pre_iter, *post_iter);
    EXPECT_EQ(&*pre_iter, &*post_iter);

    EXPECT_EQ(&*pre_iter, pre_iter.operator->());

    const char* character_address = &*pre_iter;
    abslx::Cord::CharIterator copy = pre_iter;
    ++copy;
    EXPECT_EQ(character_address, &*pre_iter);

    int n_equal_iterators = 0;
    for (abslx::Cord::CharIterator it = range.begin(); it != range.end(); ++it) {
      n_equal_iterators += static_cast<int>(it == pre_iter);
    }
    EXPECT_EQ(n_equal_iterators, 1);

    abslx::Cord::CharIterator advance_iter = range.begin();
    abslx::Cord::Advance(&advance_iter, i);
    EXPECT_EQ(pre_iter, advance_iter);

    advance_iter = range.begin();
    EXPECT_EQ(abslx::Cord::AdvanceAndRead(&advance_iter, i), cord.Subcord(0, i));
    EXPECT_EQ(pre_iter, advance_iter);

    advance_iter = pre_iter;
    abslx::Cord::Advance(&advance_iter, cord.size() - i);
    EXPECT_EQ(range.end(), advance_iter);

    advance_iter = pre_iter;
    EXPECT_EQ(abslx::Cord::AdvanceAndRead(&advance_iter, cord.size() - i),
              cord.Subcord(i, cord.size() - i));
    EXPECT_EQ(range.end(), advance_iter);

    ++i;
    ++pre_iter;
    post_iter++;
  }
  EXPECT_EQ(i, cord.size());
  EXPECT_TRUE(pre_iter == cord.char_end());   // NOLINT: explicitly test ==
  EXPECT_TRUE(post_iter == cord.char_end());  // NOLINT

  abslx::Cord::CharIterator zero_advanced_end = cord.char_end();
  abslx::Cord::Advance(&zero_advanced_end, 0);
  EXPECT_EQ(zero_advanced_end, cord.char_end());

  abslx::Cord::CharIterator it = cord.char_begin();
  for (abslx::string_view chunk : cord.Chunks()) {
    while (!chunk.empty()) {
      EXPECT_EQ(abslx::Cord::ChunkRemaining(it), chunk);
      chunk.remove_prefix(1);
      ++it;
    }
  }
}

TEST(CordCharIterator, Operations) {
  abslx::Cord empty_cord;
  VerifyCharIterator(empty_cord);

  abslx::Cord small_buffer_cord("small cord");
  VerifyCharIterator(small_buffer_cord);

  abslx::Cord flat_node_cord("larger than small buffer optimization");
  VerifyCharIterator(flat_node_cord);

  VerifyCharIterator(
      abslx::MakeFragmentedCord({"a ", "small ", "fragmented ", "cord ", "for ",
                                "testing ", "character ", "iteration."}));

  abslx::Cord reused_nodes_cord("ghi");
  reused_nodes_cord.Prepend(abslx::Cord("def"));
  reused_nodes_cord.Prepend(abslx::Cord("abc"));
  for (int i = 0; i < 4; ++i) {
    reused_nodes_cord.Prepend(reused_nodes_cord);
    VerifyCharIterator(reused_nodes_cord);
  }

  RandomEngine rng(testing::GTEST_FLAG(random_seed));
  abslx::Cord flat_cord(RandomLowercaseString(&rng, 256));
  abslx::Cord subcords;
  for (int i = 0; i < 4; ++i) subcords.Prepend(flat_cord.Subcord(16 * i, 128));
  VerifyCharIterator(subcords);
}

TEST(Cord, StreamingOutput) {
  abslx::Cord c =
      abslx::MakeFragmentedCord({"A ", "small ", "fragmented ", "Cord", "."});
  std::stringstream output;
  output << c;
  EXPECT_EQ("A small fragmented Cord.", output.str());
}

TEST(Cord, ForEachChunk) {
  for (int num_elements : {1, 10, 200}) {
    SCOPED_TRACE(num_elements);
    std::vector<std::string> cord_chunks;
    for (int i = 0; i < num_elements; ++i) {
      cord_chunks.push_back(abslx::StrCat("[", i, "]"));
    }
    abslx::Cord c = abslx::MakeFragmentedCord(cord_chunks);

    std::vector<std::string> iterated_chunks;
    abslx::CordTestPeer::ForEachChunk(c,
                                     [&iterated_chunks](abslx::string_view sv) {
                                       iterated_chunks.emplace_back(sv);
                                     });
    EXPECT_EQ(iterated_chunks, cord_chunks);
  }
}

TEST(Cord, SmallBufferAssignFromOwnData) {
  constexpr size_t kMaxInline = 15;
  std::string contents = "small buff cord";
  EXPECT_EQ(contents.size(), kMaxInline);
  for (size_t pos = 0; pos < contents.size(); ++pos) {
    for (size_t count = contents.size() - pos; count > 0; --count) {
      abslx::Cord c(contents);
      abslx::string_view flat = c.Flatten();
      c = flat.substr(pos, count);
      EXPECT_EQ(c, contents.substr(pos, count))
          << "pos = " << pos << "; count = " << count;
    }
  }
}

TEST(Cord, Format) {
  abslx::Cord c;
  abslx::Format(&c, "There were %04d little %s.", 3, "pigs");
  EXPECT_EQ(c, "There were 0003 little pigs.");
  abslx::Format(&c, "And %-3llx bad wolf!", 1);
  EXPECT_EQ(c, "There were 0003 little pigs.And 1   bad wolf!");
}

TEST(CordDeathTest, Hardening) {
  abslx::Cord cord("hello");
  // These statement should abort the program in all builds modes.
  EXPECT_DEATH_IF_SUPPORTED(cord.RemovePrefix(6), "");
  EXPECT_DEATH_IF_SUPPORTED(cord.RemoveSuffix(6), "");

  bool test_hardening = false;
  ABSL_HARDENING_ASSERT([&]() {
    // This only runs when ABSL_HARDENING_ASSERT is active.
    test_hardening = true;
    return true;
  }());
  if (!test_hardening) return;

  EXPECT_DEATH_IF_SUPPORTED(cord[5], "");
  EXPECT_DEATH_IF_SUPPORTED(*cord.chunk_end(), "");
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(cord.chunk_end()->empty()), "");
  EXPECT_DEATH_IF_SUPPORTED(++cord.chunk_end(), "");
}

class AfterExitCordTester {
 public:
  bool Set(abslx::Cord* cord, abslx::string_view expected) {
    cord_ = cord;
    expected_ = expected;
    return true;
  }

  ~AfterExitCordTester() {
    EXPECT_EQ(*cord_, expected_);
  }
 private:
  abslx::Cord* cord_;
  abslx::string_view expected_;
};

template <typename Str>
void TestConstinitConstructor(Str) {
  const auto expected = Str::value;
  // Defined before `cord` to be destroyed after it.
  static AfterExitCordTester exit_tester;  // NOLINT
  ABSL_CONST_INIT static abslx::Cord cord(Str{});  // NOLINT
  static bool init_exit_tester = exit_tester.Set(&cord, expected);
  (void)init_exit_tester;

  EXPECT_EQ(cord, expected);
  // Copy the object and test the copy, and the original.
  {
    abslx::Cord copy = cord;
    EXPECT_EQ(copy, expected);
  }
  // The original still works
  EXPECT_EQ(cord, expected);

  // Try making adding more structure to the tree.
  {
    abslx::Cord copy = cord;
    std::string expected_copy(expected);
    for (int i = 0; i < 10; ++i) {
      copy.Append(cord);
      abslx::StrAppend(&expected_copy, expected);
      EXPECT_EQ(copy, expected_copy);
    }
  }

  // Make sure we are using the right branch during constant evaluation.
  EXPECT_EQ(abslx::CordTestPeer::IsTree(cord), cord.size() >= 16);

  for (int i = 0; i < 10; ++i) {
    // Make a few more Cords from the same global rep.
    // This tests what happens when the refcount for it gets below 1.
    EXPECT_EQ(expected, abslx::Cord(Str{}));
  }
}

constexpr int SimpleStrlen(const char* p) {
  return *p ? 1 + SimpleStrlen(p + 1) : 0;
}

struct ShortView {
  constexpr abslx::string_view operator()() const {
    return abslx::string_view("SSO string", SimpleStrlen("SSO string"));
  }
};

struct LongView {
  constexpr abslx::string_view operator()() const {
    return abslx::string_view("String that does not fit SSO.",
                             SimpleStrlen("String that does not fit SSO."));
  }
};


TEST(Cord, ConstinitConstructor) {
  TestConstinitConstructor(
      abslx::strings_internal::MakeStringConstant(ShortView{}));
  TestConstinitConstructor(
      abslx::strings_internal::MakeStringConstant(LongView{}));
}
