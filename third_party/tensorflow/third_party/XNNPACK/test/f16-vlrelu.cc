// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vlrelu.yaml
//   Generator: tools/generate-vunary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vunary.h>
#include "vunary-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VLRELU__NEONFP16ARITH_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_x8, xnn_init_f16_lrelu_neon_params);
  }

  TEST(F16_VLRELU__NEONFP16ARITH_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_x8, xnn_init_f16_lrelu_neon_params);
    }
  }

  TEST(F16_VLRELU__NEONFP16ARITH_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_x8, xnn_init_f16_lrelu_neon_params);
    }
  }

  TEST(F16_VLRELU__NEONFP16ARITH_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_x8, xnn_init_f16_lrelu_neon_params);
    }
  }

  TEST(F16_VLRELU__NEONFP16ARITH_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_x8, xnn_init_f16_lrelu_neon_params);
    }
  }

  TEST(F16_VLRELU__NEONFP16ARITH_X8, slope) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (float slope : std::vector<float>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_x8, xnn_init_f16_lrelu_neon_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VLRELU__NEONFP16ARITH_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_x16, xnn_init_f16_lrelu_neon_params);
  }

  TEST(F16_VLRELU__NEONFP16ARITH_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_x16, xnn_init_f16_lrelu_neon_params);
    }
  }

  TEST(F16_VLRELU__NEONFP16ARITH_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_x16, xnn_init_f16_lrelu_neon_params);
    }
  }

  TEST(F16_VLRELU__NEONFP16ARITH_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_x16, xnn_init_f16_lrelu_neon_params);
    }
  }

  TEST(F16_VLRELU__NEONFP16ARITH_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_x16, xnn_init_f16_lrelu_neon_params);
    }
  }

  TEST(F16_VLRELU__NEONFP16ARITH_X16, slope) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (float slope : std::vector<float>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_x16, xnn_init_f16_lrelu_neon_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VLRELU__F16C_X8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vlrelu_ukernel__f16c_x8, xnn_init_f16_lrelu_avx_params);
  }

  TEST(F16_VLRELU__F16C_X8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__f16c_x8, xnn_init_f16_lrelu_avx_params);
    }
  }

  TEST(F16_VLRELU__F16C_X8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__f16c_x8, xnn_init_f16_lrelu_avx_params);
    }
  }

  TEST(F16_VLRELU__F16C_X8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__f16c_x8, xnn_init_f16_lrelu_avx_params);
    }
  }

  TEST(F16_VLRELU__F16C_X8, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vlrelu_ukernel__f16c_x8, xnn_init_f16_lrelu_avx_params);
    }
  }

  TEST(F16_VLRELU__F16C_X8, slope) {
    TEST_REQUIRES_X86_F16C;
    for (float slope : std::vector<float>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f16_vlrelu_ukernel__f16c_x8, xnn_init_f16_lrelu_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VLRELU__F16C_X16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vlrelu_ukernel__f16c_x16, xnn_init_f16_lrelu_avx_params);
  }

  TEST(F16_VLRELU__F16C_X16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__f16c_x16, xnn_init_f16_lrelu_avx_params);
    }
  }

  TEST(F16_VLRELU__F16C_X16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__f16c_x16, xnn_init_f16_lrelu_avx_params);
    }
  }

  TEST(F16_VLRELU__F16C_X16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__f16c_x16, xnn_init_f16_lrelu_avx_params);
    }
  }

  TEST(F16_VLRELU__F16C_X16, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vlrelu_ukernel__f16c_x16, xnn_init_f16_lrelu_avx_params);
    }
  }

  TEST(F16_VLRELU__F16C_X16, slope) {
    TEST_REQUIRES_X86_F16C;
    for (float slope : std::vector<float>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f16_vlrelu_ukernel__f16c_x16, xnn_init_f16_lrelu_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
