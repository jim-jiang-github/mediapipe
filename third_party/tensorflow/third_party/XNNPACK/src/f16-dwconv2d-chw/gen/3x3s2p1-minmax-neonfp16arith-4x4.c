// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv2d-chw/3x3s2p1-neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_4x4(
    size_t input_height,
    size_t input_width,
    const void* input,
    const void* weights,
    const void* zero,
    void* output,
    uint32_t padding_top,
    const union xnn_f16_chw_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(__fp16) == 0);
  assert(padding_top >= 0);
  assert(padding_top <= 1);

  const uint16x4_t vmask_even = vld1_u16(params->neonfp16arith.mask_even);
  const uint16x4_t vmask_odd  = vld1_u16(params->neonfp16arith.mask_odd);
  const float16x4_t vmax = vld1_dup_f16(&params->neonfp16arith.max);
  const float16x4_t vmin = vld1_dup_f16(&params->neonfp16arith.min);

  const __fp16* w0 = (const __fp16*)weights;
  const float16x8_t vw01234567 = vld1q_f16(w0);
  const float16x4_t vw89 = vreinterpret_f16_u32(vld1_lane_u32((const void*)(w0 + 8), vmov_n_u32(0), 0));

  const size_t input_decrement = round_down_po2(input_width, 4 /* SIMD output width */ * 2 /* subsampling */ * sizeof(__fp16));
  const size_t output_width = round_down_po2((input_width + (2 /* padding */ - 3 /* kernel size */ + 2 /* subsampling */) * sizeof(__fp16)) / 2, sizeof(__fp16));

  const __fp16* i0 = (const __fp16*) ((uintptr_t) input - ((-padding_top) & input_width));
  const __fp16* i1 = (const __fp16*) ((uintptr_t) i0 + input_width);
  if XNN_UNPREDICTABLE(padding_top != 0) {
    i0 = zero;
  }
  const __fp16* i2 = (const __fp16*) ((uintptr_t) i1 + input_width);
  const __fp16* i3 = (const __fp16*) ((uintptr_t) i2 + input_width);
  const __fp16* i4 = (const __fp16*) ((uintptr_t) i3 + input_width);
  const __fp16* i5 = (const __fp16*) ((uintptr_t) i4 + input_width);
  const __fp16* i6 = (const __fp16*) ((uintptr_t) i5 + input_width);
  const __fp16* i7 = (const __fp16*) ((uintptr_t) i6 + input_width);
  const __fp16* i8 = (const __fp16*) ((uintptr_t) i7 + input_width);

  __fp16* o0 = output;
  __fp16* o1 = (__fp16*) ((uintptr_t) o0 + output_width);
  __fp16* o2 = (__fp16*) ((uintptr_t) o1 + output_width);
  __fp16* o3 = (__fp16*) ((uintptr_t) o2 + output_width);

  size_t padded_input_height = input_height + padding_top + 1 /* padding bottom */;
  size_t output_height = (padded_input_height - 3 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 4) {
      i2 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 5) {
      i3 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i4 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 7) {
      i5 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 8) {
      i6 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 9) {
      i7 = zero;
      o3 = o2;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 10) {
      i8 = zero;
    }

    float16x4_t vi0x1357 = vmov_n_f16(0);
    float16x4_t vi1x1357 = vmov_n_f16(0);
    float16x4_t vi2x1357 = vmov_n_f16(0);
    float16x4_t vi3x1357 = vmov_n_f16(0);
    float16x4_t vi4x1357 = vmov_n_f16(0);
    float16x4_t vi5x1357 = vmov_n_f16(0);
    float16x4_t vi6x1357 = vmov_n_f16(0);
    float16x4_t vi7x1357 = vmov_n_f16(0);
    float16x4_t vi8x1357 = vmov_n_f16(0);

    size_t w = input_width;
    for (; w >= 8 * sizeof(__fp16); w -= 8 * sizeof(__fp16)) {
      float16x4_t vo0p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo1p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo2p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo3p0 = vdup_laneq_f16(vw01234567, 0);

      const float16x4x2_t vi0x8ACE9BDF = vld2_f16(i0); i0 += 8;
      const float16x4x2_t vi1x8ACE9BDF = vld2_f16(i1); i1 += 8;
      const float16x4x2_t vi2x8ACE9BDF = vld2_f16(i2); i2 += 8;
      const float16x4x2_t vi3x8ACE9BDF = vld2_f16(i3); i3 += 8;
      const float16x4x2_t vi4x8ACE9BDF = vld2_f16(i4); i4 += 8;
      const float16x4x2_t vi5x8ACE9BDF = vld2_f16(i5); i5 += 8;
      const float16x4x2_t vi6x8ACE9BDF = vld2_f16(i6); i6 += 8;
      const float16x4x2_t vi7x8ACE9BDF = vld2_f16(i7); i7 += 8;
      const float16x4x2_t vi8x8ACE9BDF = vld2_f16(i8); i8 += 8;

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x8ACE9BDF.val[0], vw01234567, 2);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x8ACE9BDF.val[0], vw01234567, 2);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x8ACE9BDF.val[0], vw01234567, 2);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x8ACE9BDF.val[0], vw01234567, 2);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x8ACE9BDF.val[0], vw01234567, 5);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x8ACE9BDF.val[0], vw01234567, 5);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x8ACE9BDF.val[0], vw01234567, 5);
      vo3p0 = vfma_laneq_f16(vo3p0, vi7x8ACE9BDF.val[0], vw01234567, 5);

      vo0p0 = vfma_lane_f16(vo0p0, vi2x8ACE9BDF.val[0], vw89, 0);
      vo1p0 = vfma_lane_f16(vo1p0, vi4x8ACE9BDF.val[0], vw89, 0);
      vo2p0 = vfma_lane_f16(vo2p0, vi6x8ACE9BDF.val[0], vw89, 0);
      vo3p0 = vfma_lane_f16(vo3p0, vi8x8ACE9BDF.val[0], vw89, 0);

      const float16x4_t vi0x7BDF = vext_f16(vi0x1357, vi0x8ACE9BDF.val[1], 3);
      vi0x1357 = vi0x8ACE9BDF.val[1];
      const float16x4_t vi1x7BDF = vext_f16(vi1x1357, vi1x8ACE9BDF.val[1], 3);
      vi1x1357 = vi1x8ACE9BDF.val[1];
      const float16x4_t vi2x7BDF = vext_f16(vi2x1357, vi2x8ACE9BDF.val[1], 3);
      vi2x1357 = vi2x8ACE9BDF.val[1];
      const float16x4_t vi3x7BDF = vext_f16(vi3x1357, vi3x8ACE9BDF.val[1], 3);
      vi3x1357 = vi3x8ACE9BDF.val[1];
      const float16x4_t vi4x7BDF = vext_f16(vi4x1357, vi4x8ACE9BDF.val[1], 3);
      vi4x1357 = vi4x8ACE9BDF.val[1];
      const float16x4_t vi5x7BDF = vext_f16(vi5x1357, vi5x8ACE9BDF.val[1], 3);
      vi5x1357 = vi5x8ACE9BDF.val[1];
      const float16x4_t vi6x7BDF = vext_f16(vi6x1357, vi6x8ACE9BDF.val[1], 3);
      vi6x1357 = vi6x8ACE9BDF.val[1];
      const float16x4_t vi7x7BDF = vext_f16(vi7x1357, vi7x8ACE9BDF.val[1], 3);
      vi7x1357 = vi7x8ACE9BDF.val[1];
      const float16x4_t vi8x7BDF = vext_f16(vi8x1357, vi8x8ACE9BDF.val[1], 3);
      vi8x1357 = vi8x8ACE9BDF.val[1];

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x7BDF, vw01234567, 1);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x7BDF, vw01234567, 1);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x7BDF, vw01234567, 1);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x7BDF, vw01234567, 1);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x7BDF, vw01234567, 4);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x7BDF, vw01234567, 4);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x7BDF, vw01234567, 4);
      vo3p0 = vfma_laneq_f16(vo3p0, vi7x7BDF, vw01234567, 4);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x7BDF, vw01234567, 7);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x7BDF, vw01234567, 7);
      vo2p0 = vfma_laneq_f16(vo2p0, vi6x7BDF, vw01234567, 7);
      vo3p0 = vfma_laneq_f16(vo3p0, vi8x7BDF, vw01234567, 7);

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x8ACE9BDF.val[1], vw01234567, 3);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x8ACE9BDF.val[1], vw01234567, 3);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x8ACE9BDF.val[1], vw01234567, 3);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x8ACE9BDF.val[1], vw01234567, 3);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x8ACE9BDF.val[1], vw01234567, 6);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x8ACE9BDF.val[1], vw01234567, 6);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x8ACE9BDF.val[1], vw01234567, 6);
      vo3p0 = vfma_laneq_f16(vo3p0, vi7x8ACE9BDF.val[1], vw01234567, 6);

      vo0p0 = vfma_lane_f16(vo0p0, vi2x8ACE9BDF.val[1], vw89, 1);
      vo1p0 = vfma_lane_f16(vo1p0, vi4x8ACE9BDF.val[1], vw89, 1);
      vo2p0 = vfma_lane_f16(vo2p0, vi6x8ACE9BDF.val[1], vw89, 1);
      vo3p0 = vfma_lane_f16(vo3p0, vi8x8ACE9BDF.val[1], vw89, 1);


      float16x4_t vo0 = vmax_f16(vo0p0, vmin);
      float16x4_t vo1 = vmax_f16(vo1p0, vmin);
      float16x4_t vo2 = vmax_f16(vo2p0, vmin);
      float16x4_t vo3 = vmax_f16(vo3p0, vmin);

      vo0 = vmin_f16(vo0, vmax);
      vo1 = vmin_f16(vo1, vmax);
      vo2 = vmin_f16(vo2, vmax);
      vo3 = vmin_f16(vo3, vmax);

      vst1_f16(o3, vo3); o3 += 4;
      vst1_f16(o2, vo2); o2 += 4;
      vst1_f16(o1, vo1); o1 += 4;
      vst1_f16(o0, vo0); o0 += 4;
    }
    // Last block has 0-7 pixels to process.
    assert(w < 8 * sizeof(__fp16));
    if XNN_LIKELY(w != 0) {
      float16x4_t vo0p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo1p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo2p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo3p0 = vdup_laneq_f16(vw01234567, 0);

      const float16x4x2_t vi0x8ACE9BDF = vld2_f16(i0);
      const float16x4x2_t vi1x8ACE9BDF = vld2_f16(i1);
      const float16x4x2_t vi2x8ACE9BDF = vld2_f16(i2);
      const float16x4x2_t vi3x8ACE9BDF = vld2_f16(i3);
      const float16x4x2_t vi4x8ACE9BDF = vld2_f16(i4);
      const float16x4x2_t vi5x8ACE9BDF = vld2_f16(i5);
      const float16x4x2_t vi6x8ACE9BDF = vld2_f16(i6);
      const float16x4x2_t vi7x8ACE9BDF = vld2_f16(i7);
      const float16x4x2_t vi8x8ACE9BDF = vld2_f16(i8);

      const float16x4_t vi0x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi0x8ACE9BDF.val[0])));
      const float16x4_t vi0x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd,  vreinterpret_u16_f16(vi0x8ACE9BDF.val[1])));
      const float16x4_t vi1x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi1x8ACE9BDF.val[0])));
      const float16x4_t vi1x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd,  vreinterpret_u16_f16(vi1x8ACE9BDF.val[1])));
      const float16x4_t vi2x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi2x8ACE9BDF.val[0])));
      const float16x4_t vi2x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd,  vreinterpret_u16_f16(vi2x8ACE9BDF.val[1])));
      const float16x4_t vi3x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi3x8ACE9BDF.val[0])));
      const float16x4_t vi3x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd,  vreinterpret_u16_f16(vi3x8ACE9BDF.val[1])));
      const float16x4_t vi4x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi4x8ACE9BDF.val[0])));
      const float16x4_t vi4x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd,  vreinterpret_u16_f16(vi4x8ACE9BDF.val[1])));
      const float16x4_t vi5x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi5x8ACE9BDF.val[0])));
      const float16x4_t vi5x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd,  vreinterpret_u16_f16(vi5x8ACE9BDF.val[1])));
      const float16x4_t vi6x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi6x8ACE9BDF.val[0])));
      const float16x4_t vi6x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd,  vreinterpret_u16_f16(vi6x8ACE9BDF.val[1])));
      const float16x4_t vi7x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi7x8ACE9BDF.val[0])));
      const float16x4_t vi7x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd,  vreinterpret_u16_f16(vi7x8ACE9BDF.val[1])));
      const float16x4_t vi8x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi8x8ACE9BDF.val[0])));
      const float16x4_t vi8x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd,  vreinterpret_u16_f16(vi8x8ACE9BDF.val[1])));

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x8ACE, vw01234567, 2);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x8ACE, vw01234567, 2);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x8ACE, vw01234567, 2);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x8ACE, vw01234567, 2);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x8ACE, vw01234567, 5);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x8ACE, vw01234567, 5);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x8ACE, vw01234567, 5);
      vo3p0 = vfma_laneq_f16(vo3p0, vi7x8ACE, vw01234567, 5);

      vo0p0 = vfma_lane_f16(vo0p0, vi2x8ACE, vw89, 0);
      vo1p0 = vfma_lane_f16(vo1p0, vi4x8ACE, vw89, 0);
      vo2p0 = vfma_lane_f16(vo2p0, vi6x8ACE, vw89, 0);
      vo3p0 = vfma_lane_f16(vo3p0, vi8x8ACE, vw89, 0);

      const float16x4_t vi0x7BDF = vext_f16(vi0x1357, vi0x9BDF, 3);
      const float16x4_t vi1x7BDF = vext_f16(vi1x1357, vi1x9BDF, 3);
      const float16x4_t vi2x7BDF = vext_f16(vi2x1357, vi2x9BDF, 3);
      const float16x4_t vi3x7BDF = vext_f16(vi3x1357, vi3x9BDF, 3);
      const float16x4_t vi4x7BDF = vext_f16(vi4x1357, vi4x9BDF, 3);
      const float16x4_t vi5x7BDF = vext_f16(vi5x1357, vi5x9BDF, 3);
      const float16x4_t vi6x7BDF = vext_f16(vi6x1357, vi6x9BDF, 3);
      const float16x4_t vi7x7BDF = vext_f16(vi7x1357, vi7x9BDF, 3);
      const float16x4_t vi8x7BDF = vext_f16(vi8x1357, vi8x9BDF, 3);

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x7BDF, vw01234567, 1);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x7BDF, vw01234567, 1);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x7BDF, vw01234567, 1);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x7BDF, vw01234567, 1);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x7BDF, vw01234567, 4);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x7BDF, vw01234567, 4);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x7BDF, vw01234567, 4);
      vo3p0 = vfma_laneq_f16(vo3p0, vi7x7BDF, vw01234567, 4);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x7BDF, vw01234567, 7);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x7BDF, vw01234567, 7);
      vo2p0 = vfma_laneq_f16(vo2p0, vi6x7BDF, vw01234567, 7);
      vo3p0 = vfma_laneq_f16(vo3p0, vi8x7BDF, vw01234567, 7);

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x9BDF, vw01234567, 3);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x9BDF, vw01234567, 3);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x9BDF, vw01234567, 3);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x9BDF, vw01234567, 3);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x9BDF, vw01234567, 6);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x9BDF, vw01234567, 6);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x9BDF, vw01234567, 6);
      vo3p0 = vfma_laneq_f16(vo3p0, vi7x9BDF, vw01234567, 6);

      vo0p0 = vfma_lane_f16(vo0p0, vi2x9BDF, vw89, 1);
      vo1p0 = vfma_lane_f16(vo1p0, vi4x9BDF, vw89, 1);
      vo2p0 = vfma_lane_f16(vo2p0, vi6x9BDF, vw89, 1);
      vo3p0 = vfma_lane_f16(vo3p0, vi8x9BDF, vw89, 1);


      float16x4_t vo0 = vmax_f16(vo0p0, vmin);
      float16x4_t vo1 = vmax_f16(vo1p0, vmin);
      float16x4_t vo2 = vmax_f16(vo2p0, vmin);
      float16x4_t vo3 = vmax_f16(vo3p0, vmin);

      vo0 = vmin_f16(vo0, vmax);
      vo1 = vmin_f16(vo1, vmax);
      vo2 = vmin_f16(vo2, vmax);
      vo3 = vmin_f16(vo3, vmax);

      w += 1 * sizeof(__fp16);

      if XNN_LIKELY(w == 8 * sizeof(__fp16)) {
        vst1_f16(o3, vo3); o3 += 4;
        vst1_f16(o2, vo2); o2 += 4;
        vst1_f16(o1, vo1); o1 += 4;
        vst1_f16(o0, vo0); o0 += 4;
      } else {
        if (w & (4 * sizeof(__fp16))) {
          vst1_lane_u32((void*) o3, vreinterpret_u32_f16(vo3), 0); o3 += 2;
          vst1_lane_u32((void*) o2, vreinterpret_u32_f16(vo2), 0); o2 += 2;
          vst1_lane_u32((void*) o1, vreinterpret_u32_f16(vo1), 0); o1 += 2;
          vst1_lane_u32((void*) o0, vreinterpret_u32_f16(vo0), 0); o0 += 2;

          vo0 = vext_f16(vo0, vo0, 2);
          vo1 = vext_f16(vo1, vo1, 2);
          vo2 = vext_f16(vo2, vo2, 2);
          vo3 = vext_f16(vo3, vo3, 2);
        }
        if (w & (2 * sizeof(__fp16))) {
          vst1_lane_f16(o3, vo3, 0); o3 += 1;
          vst1_lane_f16(o2, vo2, 0); o2 += 1;
          vst1_lane_f16(o1, vo1, 0); o1 += 1;
          vst1_lane_f16(o0, vo0, 0); o0 += 1;
        }
      }

    }

    i0 = (const __fp16*) ((uintptr_t) i8 - input_decrement);
    i1 = (const __fp16*) ((uintptr_t) i0 + input_width);
    i2 = (const __fp16*) ((uintptr_t) i1 + input_width);
    i3 = (const __fp16*) ((uintptr_t) i2 + input_width);
    i4 = (const __fp16*) ((uintptr_t) i3 + input_width);
    i5 = (const __fp16*) ((uintptr_t) i4 + input_width);
    i6 = (const __fp16*) ((uintptr_t) i5 + input_width);
    i7 = (const __fp16*) ((uintptr_t) i6 + input_width);
    i8 = (const __fp16*) ((uintptr_t) i7 + input_width);

    o0 = o3;
    o1 = (__fp16*) ((uintptr_t) o0 + output_width);
    o2 = (__fp16*) ((uintptr_t) o1 + output_width);
    o3 = (__fp16*) ((uintptr_t) o2 + output_width);

    output_height = doz(output_height, 4);
    padded_input_height = doz(padded_input_height, 8);
  } while (output_height != 0);
}
