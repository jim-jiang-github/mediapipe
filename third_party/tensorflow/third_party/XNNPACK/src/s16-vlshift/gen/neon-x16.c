// Auto-generated file. Do not edit!
//   Template: src/s16-vlshift/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include <xnnpack/math.h>
#include <xnnpack/vlshift.h>


void xnn_s16_vlshift_ukernel__neon_x16(
    size_t batch,
    const int16_t* input,
    uint32_t shift,
    int16_t* output) {

  assert(batch > 0);
  assert(input != NULL);
  assert(shift < 16);
  assert(output != NULL);

  const int16x8_t vshift = vdupq_n_s16(shift);

  for (; batch >= 16; batch -= 16) {
    const int16x8_t vi0 = vld1q_s16(input); input += 8;
    const int16x8_t vi1 = vld1q_s16(input); input += 8;

    const int16x8_t vout0 = vshlq_s16(vi0, vshift);
    const int16x8_t vout1 = vshlq_s16(vi1, vshift);

    vst1q_s16(output, vout0); output += 8;
    vst1q_s16(output, vout1); output += 8;
  }

  // Remainder of full vectors
  for (; batch >= 8; batch -= 8) {
    const int16x8_t vi = vld1q_s16(input); input += 8;

    const int16x8_t vout = vshlq_s16(vi, vshift);

    vst1q_s16(output, vout); output += 8;
  }

  // Remainder of 1 to 7 batch
  if XNN_UNLIKELY(batch != 0) {
    const int16x8_t vi = vld1q_s16(input);

    const int16x8_t vout = vshlq_s16(vi, vshift);
    int16x4_t vout_lo = vget_low_s16(vout);

    if (batch & 4) {
      vst1_s16(output, vout_lo); output += 4;
      vout_lo = vget_high_s16(vout);
    }
    if (batch & 2) {
      vst1_lane_u32((void*) output, vreinterpret_u32_s16(vout_lo), 0); output += 2;
      vout_lo = vext_s16(vout_lo, vout_lo, 2);
    }
    if (batch & 1){
      vst1_lane_s16(output, vout_lo, 0);
    }
  }
}
