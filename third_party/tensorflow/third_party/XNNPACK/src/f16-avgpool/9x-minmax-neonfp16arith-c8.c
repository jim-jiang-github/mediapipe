// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/avgpool.h>


void xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    const void* zero,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);

  const float16x8_t vscale = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neon.scale));
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neon.min));
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neon.max));

  do {
    const __fp16* i0 = (const __fp16*) input[0];
    assert(i0 != NULL);
    const __fp16* i1 = (const __fp16*) input[1];
    const __fp16* i2 = (const __fp16*) input[2];
    const __fp16* i3 = (const __fp16*) input[3];
    const __fp16* i4 = (const __fp16*) input[4];
    const __fp16* i5 = (const __fp16*) input[5];
    const __fp16* i6 = (const __fp16*) input[6];
    const __fp16* i7 = (const __fp16*) input[7];
    const __fp16* i8 = (const __fp16*) input[8];
    input = (const void**) ((uintptr_t) input + input_increment);
    if (kernel_elements < 2) {
      i1 = (const __fp16*) zero;
    }
    assert(i1 != NULL);
    if (kernel_elements <= 2) {
      i2 = (const __fp16*) zero;
    }
    assert(i2 != NULL);
    if (kernel_elements < 4) {
      i3 = (const __fp16*) zero;
    }
    assert(i3 != NULL);
    if (kernel_elements <= 4) {
      i4 = (const __fp16*) zero;
    }
    assert(i4 != NULL);
    if (kernel_elements < 6) {
      i5 = (const __fp16*) zero;
    }
    assert(i5 != NULL);
    if (kernel_elements <= 6) {
      i6 = (const __fp16*) zero;
    }
    assert(i6 != NULL);
    if (kernel_elements < 8) {
      i7 = (const __fp16*) zero;
    }
    assert(i7 != NULL);
    if (kernel_elements <= 8) {
      i8 = (const __fp16*) zero;
    }
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const __fp16*) ((uintptr_t) i0 + input_offset);
    }
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const __fp16*) ((uintptr_t) i1 + input_offset);
    }
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const __fp16*) ((uintptr_t) i2 + input_offset);
    }
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const __fp16*) ((uintptr_t) i3 + input_offset);
    }
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const __fp16*) ((uintptr_t) i4 + input_offset);
    }
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const __fp16*) ((uintptr_t) i5 + input_offset);
    }
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const __fp16*) ((uintptr_t) i6 + input_offset);
    }
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const __fp16*) ((uintptr_t) i7 + input_offset);
    }
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const __fp16*) ((uintptr_t) i8 + input_offset);
    }

    size_t c = channels;
    while (c >= 8) {
      const float16x8_t vi0 = vld1q_f16(i0); i0 += 8;
      const float16x8_t vi1 = vld1q_f16(i1); i1 += 8;
      const float16x8_t vi2 = vld1q_f16(i2); i2 += 8;
      const float16x8_t vi3 = vld1q_f16(i3); i3 += 8;
      const float16x8_t vi4 = vld1q_f16(i4); i4 += 8;
      const float16x8_t vi5 = vld1q_f16(i5); i5 += 8;
      const float16x8_t vi6 = vld1q_f16(i6); i6 += 8;
      const float16x8_t vi7 = vld1q_f16(i7); i7 += 8;
      const float16x8_t vi8 = vld1q_f16(i8); i8 += 8;

      const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
      const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
      const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
      const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
      const float16x8_t vsum018 = vaddq_f16(vsum01, vi8);
      const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
      const float16x8_t vsum01678 = vaddq_f16(vsum018, vsum67);
      const float16x8_t vsum = vaddq_f16(vsum2345, vsum01678);

      float16x8_t vout = vmulq_f16(vsum, vscale);
      vout = vmaxq_f16(vout, vmin);
      vout = vminq_f16(vout, vmax);

      vst1q_f16(output, vout); output = (__fp16*) output + 8;

      c -= 8;
    }
    if (c != 0) {
      const float16x8_t vi0 = vld1q_f16(i0);
      const float16x8_t vi1 = vld1q_f16(i1);
      const float16x8_t vi2 = vld1q_f16(i2);
      const float16x8_t vi3 = vld1q_f16(i3);
      const float16x8_t vi4 = vld1q_f16(i4);
      const float16x8_t vi5 = vld1q_f16(i5);
      const float16x8_t vi6 = vld1q_f16(i6);
      const float16x8_t vi7 = vld1q_f16(i7);
      const float16x8_t vi8 = vld1q_f16(i8);

      const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
      const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
      const float16x8_t vsum45 = vaddq_f16(vi4, vi5);
      const float16x8_t vsum67 = vaddq_f16(vi6, vi7);
      const float16x8_t vsum018 = vaddq_f16(vsum01, vi8);
      const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);
      const float16x8_t vsum01678 = vaddq_f16(vsum018, vsum67);
      const float16x8_t vsum = vaddq_f16(vsum2345, vsum01678);

      float16x8_t vout = vmulq_f16(vsum, vscale);
      vout = vmaxq_f16(vout, vmin);
      vout = vminq_f16(vout, vmax);

      float16x4_t vout_lo = vget_low_f16(vout);
      if (c & 4) {
        vst1_f16(output, vout_lo); output = (__fp16*) output + 4;
        vout_lo = vget_high_f16(vout);
      }
      if (c & 2) {
        vst1_lane_u32(output, vreinterpret_u32_f16(vout_lo), 0); output = (__fp16*) output + 2;
        vout_lo = vext_f16(vout_lo, vout_lo, 2);
      }
      if (c & 1) {
        vst1_lane_f16(output, vout_lo, 0); output = (__fp16*) output + 1;
      }
    }
    output = (__fp16*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
