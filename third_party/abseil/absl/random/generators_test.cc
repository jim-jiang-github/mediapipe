// Copyright 2017 The Abseil Authors.
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

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"

namespace {

template <typename URBG>
void TestUniform(URBG* gen) {
  // [a, b) default-semantics, inferred types.
  abslx::Uniform(*gen, 0, 100);     // int
  abslx::Uniform(*gen, 0, 1.0);     // Promoted to double
  abslx::Uniform(*gen, 0.0f, 1.0);  // Promoted to double
  abslx::Uniform(*gen, 0.0, 1.0);   // double
  abslx::Uniform(*gen, -1, 1L);     // Promoted to long

  // Roll a die.
  abslx::Uniform(abslx::IntervalClosedClosed, *gen, 1, 6);

  // Get a fraction.
  abslx::Uniform(abslx::IntervalOpenOpen, *gen, 0.0, 1.0);

  // Assign a value to a random element.
  std::vector<int> elems = {10, 20, 30, 40, 50};
  elems[abslx::Uniform(*gen, 0u, elems.size())] = 5;
  elems[abslx::Uniform<size_t>(*gen, 0, elems.size())] = 3;

  // Choose some epsilon around zero.
  abslx::Uniform(abslx::IntervalOpenOpen, *gen, -1.0, 1.0);

  // (a, b) semantics, inferred types.
  abslx::Uniform(abslx::IntervalOpenOpen, *gen, 0, 1.0);  // Promoted to double

  // Explict overriding of types.
  abslx::Uniform<int>(*gen, 0, 100);
  abslx::Uniform<int8_t>(*gen, 0, 100);
  abslx::Uniform<int16_t>(*gen, 0, 100);
  abslx::Uniform<uint16_t>(*gen, 0, 100);
  abslx::Uniform<int32_t>(*gen, 0, 1 << 10);
  abslx::Uniform<uint32_t>(*gen, 0, 1 << 10);
  abslx::Uniform<int64_t>(*gen, 0, 1 << 10);
  abslx::Uniform<uint64_t>(*gen, 0, 1 << 10);

  abslx::Uniform<float>(*gen, 0.0, 1.0);
  abslx::Uniform<float>(*gen, 0, 1);
  abslx::Uniform<float>(*gen, -1, 1);
  abslx::Uniform<double>(*gen, 0.0, 1.0);

  abslx::Uniform<float>(*gen, -1.0, 0);
  abslx::Uniform<double>(*gen, -1.0, 0);

  // Tagged
  abslx::Uniform<double>(abslx::IntervalClosedClosed, *gen, 0, 1);
  abslx::Uniform<double>(abslx::IntervalClosedOpen, *gen, 0, 1);
  abslx::Uniform<double>(abslx::IntervalOpenOpen, *gen, 0, 1);
  abslx::Uniform<double>(abslx::IntervalOpenClosed, *gen, 0, 1);
  abslx::Uniform<double>(abslx::IntervalClosedClosed, *gen, 0, 1);
  abslx::Uniform<double>(abslx::IntervalOpenOpen, *gen, 0, 1);

  abslx::Uniform<int>(abslx::IntervalClosedClosed, *gen, 0, 100);
  abslx::Uniform<int>(abslx::IntervalClosedOpen, *gen, 0, 100);
  abslx::Uniform<int>(abslx::IntervalOpenOpen, *gen, 0, 100);
  abslx::Uniform<int>(abslx::IntervalOpenClosed, *gen, 0, 100);
  abslx::Uniform<int>(abslx::IntervalClosedClosed, *gen, 0, 100);
  abslx::Uniform<int>(abslx::IntervalOpenOpen, *gen, 0, 100);

  // With *generator as an R-value reference.
  abslx::Uniform<int>(URBG(), 0, 100);
  abslx::Uniform<double>(URBG(), 0.0, 1.0);
}

template <typename URBG>
void TestExponential(URBG* gen) {
  abslx::Exponential<float>(*gen);
  abslx::Exponential<double>(*gen);
  abslx::Exponential<double>(URBG());
}

template <typename URBG>
void TestPoisson(URBG* gen) {
  // [rand.dist.pois] Indicates that the std::poisson_distribution
  // is parameterized by IntType, however MSVC does not allow 8-bit
  // types.
  abslx::Poisson<int>(*gen);
  abslx::Poisson<int16_t>(*gen);
  abslx::Poisson<uint16_t>(*gen);
  abslx::Poisson<int32_t>(*gen);
  abslx::Poisson<uint32_t>(*gen);
  abslx::Poisson<int64_t>(*gen);
  abslx::Poisson<uint64_t>(*gen);
  abslx::Poisson<uint64_t>(URBG());
}

template <typename URBG>
void TestBernoulli(URBG* gen) {
  abslx::Bernoulli(*gen, 0.5);
  abslx::Bernoulli(*gen, 0.5);
}

template <typename URBG>
void TestZipf(URBG* gen) {
  abslx::Zipf<int>(*gen, 100);
  abslx::Zipf<int8_t>(*gen, 100);
  abslx::Zipf<int16_t>(*gen, 100);
  abslx::Zipf<uint16_t>(*gen, 100);
  abslx::Zipf<int32_t>(*gen, 1 << 10);
  abslx::Zipf<uint32_t>(*gen, 1 << 10);
  abslx::Zipf<int64_t>(*gen, 1 << 10);
  abslx::Zipf<uint64_t>(*gen, 1 << 10);
  abslx::Zipf<uint64_t>(URBG(), 1 << 10);
}

template <typename URBG>
void TestGaussian(URBG* gen) {
  abslx::Gaussian<float>(*gen, 1.0, 1.0);
  abslx::Gaussian<double>(*gen, 1.0, 1.0);
  abslx::Gaussian<double>(URBG(), 1.0, 1.0);
}

template <typename URBG>
void TestLogNormal(URBG* gen) {
  abslx::LogUniform<int>(*gen, 0, 100);
  abslx::LogUniform<int8_t>(*gen, 0, 100);
  abslx::LogUniform<int16_t>(*gen, 0, 100);
  abslx::LogUniform<uint16_t>(*gen, 0, 100);
  abslx::LogUniform<int32_t>(*gen, 0, 1 << 10);
  abslx::LogUniform<uint32_t>(*gen, 0, 1 << 10);
  abslx::LogUniform<int64_t>(*gen, 0, 1 << 10);
  abslx::LogUniform<uint64_t>(*gen, 0, 1 << 10);
  abslx::LogUniform<uint64_t>(URBG(), 0, 1 << 10);
}

template <typename URBG>
void CompatibilityTest() {
  URBG gen;

  TestUniform(&gen);
  TestExponential(&gen);
  TestPoisson(&gen);
  TestBernoulli(&gen);
  TestZipf(&gen);
  TestGaussian(&gen);
  TestLogNormal(&gen);
}

TEST(std_mt19937_64, Compatibility) {
  // Validate with std::mt19937_64
  CompatibilityTest<std::mt19937_64>();
}

TEST(BitGen, Compatibility) {
  // Validate with abslx::BitGen
  CompatibilityTest<abslx::BitGen>();
}

TEST(InsecureBitGen, Compatibility) {
  // Validate with abslx::InsecureBitGen
  CompatibilityTest<abslx::InsecureBitGen>();
}

}  // namespace
