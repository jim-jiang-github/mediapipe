// Auto-generated file. Do not edit!
//   Template: src/qs8-vlrelu/armv6simd.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_acle.h>

#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/unaligned.h>
#include <xnnpack/vlrelu.h>


void xnn_qs8_vlrelu_ukernel__armv6simd_x8(
    size_t n,
    const int8_t* x,
    int8_t* y,
    const union xnn_qs8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  const int16x2_t vinput_zero_point = (int16x2_t) params->armv6simd.input_zero_point;
  const int16x2_t vpositive_multiplier = (int16x2_t) params->armv6simd.positive_multiplier;
  const int16x2_t vnegative_multiplier = (int16x2_t) params->armv6simd.negative_multiplier;
  const int32_t vbias = params->armv6simd.bias;
  for (; n >= 8 * sizeof(int8_t); n -= 8 * sizeof(int8_t)) {
    const int8x4_t vx0123 = (int8x4_t) unaligned_indexed_load_u32(x, 0);
    const int8x4_t vx4567 = (int8x4_t) unaligned_indexed_load_u32(x, 1);
    x += 8;

    int16x2_t vx02 = __sxtb16(vx0123);
    int16x2_t vx13 = __sxtb16(__ror(vx0123, 8));
    int16x2_t vx46 = __sxtb16(vx4567);
    int16x2_t vx57 = __sxtb16(__ror(vx4567, 8));

    vx02 = __ssub16(vinput_zero_point, vx02);
    const int16x2_t vmultiplier02 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);
    vx13 = __ssub16(vinput_zero_point, vx13);
    const int16x2_t vmultiplier13 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);
    vx46 = __ssub16(vinput_zero_point, vx46);
    const int16x2_t vmultiplier46 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);
    vx57 = __ssub16(vinput_zero_point, vx57);
    const int16x2_t vmultiplier57 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);

    int32_t vacc0 = __smlabb(vmultiplier02, vx02, vbias);
    int32_t vacc1 = __smlabb(vmultiplier13, vx13, vbias);
    int32_t vacc2 = __smlatt(vmultiplier02, vx02, vbias);
    int32_t vacc3 = __smlatt(vmultiplier13, vx13, vbias);
    int32_t vacc4 = __smlabb(vmultiplier46, vx46, vbias);
    int32_t vacc5 = __smlabb(vmultiplier57, vx57, vbias);
    int32_t vacc6 = __smlatt(vmultiplier46, vx46, vbias);
    int32_t vacc7 = __smlatt(vmultiplier57, vx57, vbias);

    vacc0 = __ssat(math_asr_s32(vacc0, 8), 8);
    vacc1 = __ssat(math_asr_s32(vacc1, 8), 8);
    vacc2 = __ssat(math_asr_s32(vacc2, 8), 8);
    vacc3 = __ssat(math_asr_s32(vacc3, 8), 8);
    vacc4 = __ssat(math_asr_s32(vacc4, 8), 8);
    vacc5 = __ssat(math_asr_s32(vacc5, 8), 8);
    vacc6 = __ssat(math_asr_s32(vacc6, 8), 8);
    vacc7 = __ssat(math_asr_s32(vacc7, 8), 8);

    y[0] = (int8_t) vacc0;
    y[1] = (int8_t) vacc1;
    y[2] = (int8_t) vacc2;
    y[3] = (int8_t) vacc3;
    y[4] = (int8_t) vacc4;
    y[5] = (int8_t) vacc5;
    y[6] = (int8_t) vacc6;
    y[7] = (int8_t) vacc7;
    y += 8;
  }
  for (; n >= 4 * sizeof(int8_t); n -= 4 * sizeof(int8_t)) {
    const int8x4_t vx0123 = (int8x4_t) unaligned_load_u32(x);
    x += 4;

    int16x2_t vx02 = __sxtb16(vx0123);
    int16x2_t vx13 = __sxtb16(__ror(vx0123, 8));

    vx02 = __ssub16(vinput_zero_point, vx02);
    const int16x2_t vmultiplier02 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);
    vx13 = __ssub16(vinput_zero_point, vx13);
    const int16x2_t vmultiplier13 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);

    int32_t vacc0 = __smlabb(vmultiplier02, vx02, vbias);
    int32_t vacc1 = __smlabb(vmultiplier13, vx13, vbias);
    int32_t vacc2 = __smlatt(vmultiplier02, vx02, vbias);
    int32_t vacc3 = __smlatt(vmultiplier13, vx13, vbias);

    vacc0 = __ssat(math_asr_s32(vacc0, 8), 8);
    vacc1 = __ssat(math_asr_s32(vacc1, 8), 8);
    vacc2 = __ssat(math_asr_s32(vacc2, 8), 8);
    vacc3 = __ssat(math_asr_s32(vacc3, 8), 8);

    y[0] = (int8_t) vacc0;
    y[1] = (int8_t) vacc1;
    y[2] = (int8_t) vacc2;
    y[3] = (int8_t) vacc3;
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const int8x4_t vx0123 = (int8x4_t) unaligned_load_u32(x);

    int16x2_t vx02 = __sxtb16(vx0123);
    int16x2_t vx13 = __sxtb16(__ror(vx0123, 8));

    vx02 = __ssub16(vinput_zero_point, vx02);
    const int16x2_t vmultiplier02 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);
    vx13 = __ssub16(vinput_zero_point, vx13);
    const int16x2_t vmultiplier13 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);

    int32_t vacc0 = __smlabb(vmultiplier02, vx02, vbias);
    int32_t vacc1 = __smlabb(vmultiplier13, vx13, vbias);
    const int32_t vacc2 = __smlatt(vmultiplier02, vx02, vbias);

    vacc0 = __ssat(math_asr_s32(vacc0, 8), 8);
    vacc1 = __ssat(math_asr_s32(vacc1, 8), 8);

    if (n & (2 * sizeof(int8_t))) {
      y[0] = (int8_t) vacc0;
      y[1] = (int8_t) vacc1;
      vacc0 = __ssat(math_asr_s32(vacc2, 8), 8);
      y += 2;
    }
    if (n & (1 * sizeof(int8_t))) {
      y[0] = (int8_t) vacc0;
    }
  }
}
