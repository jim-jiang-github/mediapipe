// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/neon-mlal-lane.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/igemm.h>


void xnn_qs8_igemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
    int32x4_t vacc0x89AB = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
    int32x4_t vacc0xCDEF = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc1x89AB = vacc0x89AB;
    int32x4_t vacc1xCDEF = vacc0xCDEF;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;
    int32x4_t vacc2x89AB = vacc0x89AB;
    int32x4_t vacc2xCDEF = vacc0xCDEF;
    int32x4_t vacc3x0123 = vacc0x0123;
    int32x4_t vacc3x4567 = vacc0x4567;
    int32x4_t vacc3x89AB = vacc0x89AB;
    int32x4_t vacc3xCDEF = vacc0xCDEF;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      while (k >= 8 * sizeof(int8_t)) {
        const int8x8_t va0 = vld1_s8(a0); a0 += 8;
        const int16x8_t vxa0 = vmovl_s8(va0);
        const int8x8_t va1 = vld1_s8(a1); a1 += 8;
        const int16x8_t vxa1 = vmovl_s8(va1);
        const int8x8_t va2 = vld1_s8(a2); a2 += 8;
        const int16x8_t vxa2 = vmovl_s8(va2);
        const int8x8_t va3 = vld1_s8(a3); a3 += 8;
        const int16x8_t vxa3 = vmovl_s8(va3);

        const int8x8_t vb01234567c0 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
        const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
        const int8x8_t vb89ABCDEFc0 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
        const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);
        const int8x8_t vb01234567c1 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
        const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
        const int8x8_t vb89ABCDEFc1 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
        const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);
        const int8x8_t vb01234567c2 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
        const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
        const int8x8_t vb89ABCDEFc2 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
        const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);
        const int8x8_t vb01234567c3 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
        const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
        const int8x8_t vb89ABCDEFc3 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
        const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);

        __builtin_prefetch((const int8_t*) w + 480);
        __builtin_prefetch((const int8_t*) w + 544);

        const int8x8_t vb01234567c4 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
        const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
        const int8x8_t vb89ABCDEFc4 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
        const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);
        const int8x8_t vb01234567c5 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
        const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
        const int8x8_t vb89ABCDEFc5 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
        const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);
        const int8x8_t vb01234567c6 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
        const int16x8_t vxb01234567c6 = vmovl_s8(vb01234567c6);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
        const int8x8_t vb89ABCDEFc6 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
        const int16x8_t vxb89ABCDEFc6 = vmovl_s8(vb89ABCDEFc6);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa1), 2);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa1), 2);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa2), 2);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa2), 2);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa3), 2);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa3), 2);
        const int8x8_t vb01234567c7 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
        const int16x8_t vxb01234567c7 = vmovl_s8(vb01234567c7);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
        const int8x8_t vb89ABCDEFc7 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
        const int16x8_t vxb89ABCDEFc7 = vmovl_s8(vb89ABCDEFc7);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa1), 3);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa1), 3);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa2), 3);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa2), 3);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa3), 3);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa3), 3);

        k -= 8 * sizeof(int8_t);
      }
      if XNN_UNLIKELY(k != 0) {
        const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
        const int16x8_t vxa0 = vmovl_s8(va0);
        const int8x8_t va1 = vld1_s8(a1); a1 = (const int8_t*) ((uintptr_t) a1 + k);
        const int16x8_t vxa1 = vmovl_s8(va1);
        const int8x8_t va2 = vld1_s8(a2); a2 = (const int8_t*) ((uintptr_t) a2 + k);
        const int16x8_t vxa2 = vmovl_s8(va2);
        const int8x8_t va3 = vld1_s8(a3); a3 = (const int8_t*) ((uintptr_t) a3 + k);
        const int16x8_t vxa3 = vmovl_s8(va3);

        const int8x8_t vb01234567c0 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
        const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);
        const int8x8_t vb89ABCDEFc0 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
        const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);

        if (k >= 2 * sizeof(int8_t)) {
          const int8x8_t vb01234567c1 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
          const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);
          const int8x8_t vb89ABCDEFc1 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
          const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);

          vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
          vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
          vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
          vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
          vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
          vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
          vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
          vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
          vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
          vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
          vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
          vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
          vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
          vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
          vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);
          vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);

          if (k > 2 * sizeof(int8_t)) {
            const int8x8_t vb01234567c2 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
            const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);
            const int8x8_t vb89ABCDEFc2 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
            const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
            vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
            vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
            vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
            vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
            vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
            vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
            vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
            vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
            vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
            vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
            vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
            vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
            vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);
            vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);

            if (k >= 4 * sizeof(int8_t)) {
              const int8x8_t vb01234567c3 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
              const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);
              const int8x8_t vb89ABCDEFc3 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
              const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);

              vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
              vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
              vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
              vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
              vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
              vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
              vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
              vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
              vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
              vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
              vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
              vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
              vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
              vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
              vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);
              vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);

              if (k > 4 * sizeof(int8_t)) {
                const int8x8_t vb01234567c4 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
                const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);
                const int8x8_t vb89ABCDEFc4 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
                const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);

                vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
                vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
                vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
                vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
                vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
                vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
                vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
                vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
                vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
                vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
                vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
                vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
                vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
                vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
                vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);
                vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);

                if (k >= 6 * sizeof(int8_t)) {
                  const int8x8_t vb01234567c5 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
                  const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);
                  const int8x8_t vb89ABCDEFc5 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
                  const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);

                  vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                  vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                  vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
                  vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
                  vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
                  vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
                  vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
                  vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
                  vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
                  vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
                  vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
                  vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
                  vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
                  vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
                  vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);
                  vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);

                  if (k > 6 * sizeof(int8_t)) {
                    const int8x8_t vb01234567c6 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
                    const int16x8_t vxb01234567c6 = vmovl_s8(vb01234567c6);
                    const int8x8_t vb89ABCDEFc6 = vld1_s8(w); w = (const void*) ((const int8_t*) w + 8);
                    const int16x8_t vxb89ABCDEFc6 = vmovl_s8(vb89ABCDEFc6);

                    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
                    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
                    vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
                    vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
                    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
                    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
                    vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa1), 2);
                    vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa1), 2);
                    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
                    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
                    vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa2), 2);
                    vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa2), 2);
                    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
                    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
                    vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa3), 2);
                    vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa3), 2);
                  }
                }
              }
            }
          }
        }
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    // Post-accumulation work
    const int32x4_t vright_pre_shift = vld1q_dup_s32(&params->rndnu_neon.right_pre_shift);
    const int32x4_t vmultiplier = vld1q_dup_s32(&params->rndnu_neon.multiplier);
    const int32x4_t vright_post_shift = vld1q_dup_s32(&params->rndnu_neon.right_post_shift);

    vacc0x0123 = vshlq_s32(vacc0x0123, vright_pre_shift);
    vacc0x4567 = vshlq_s32(vacc0x4567, vright_pre_shift);
    vacc0x89AB = vshlq_s32(vacc0x89AB, vright_pre_shift);
    vacc0xCDEF = vshlq_s32(vacc0xCDEF, vright_pre_shift);
    vacc1x0123 = vshlq_s32(vacc1x0123, vright_pre_shift);
    vacc1x4567 = vshlq_s32(vacc1x4567, vright_pre_shift);
    vacc1x89AB = vshlq_s32(vacc1x89AB, vright_pre_shift);
    vacc1xCDEF = vshlq_s32(vacc1xCDEF, vright_pre_shift);
    vacc2x0123 = vshlq_s32(vacc2x0123, vright_pre_shift);
    vacc2x4567 = vshlq_s32(vacc2x4567, vright_pre_shift);
    vacc2x89AB = vshlq_s32(vacc2x89AB, vright_pre_shift);
    vacc2xCDEF = vshlq_s32(vacc2xCDEF, vright_pre_shift);
    vacc3x0123 = vshlq_s32(vacc3x0123, vright_pre_shift);
    vacc3x4567 = vshlq_s32(vacc3x4567, vright_pre_shift);
    vacc3x89AB = vshlq_s32(vacc3x89AB, vright_pre_shift);
    vacc3xCDEF = vshlq_s32(vacc3xCDEF, vright_pre_shift);

    vacc0x0123 = vqdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqdmulhq_s32(vacc0x4567, vmultiplier);
    vacc0x89AB = vqdmulhq_s32(vacc0x89AB, vmultiplier);
    vacc0xCDEF = vqdmulhq_s32(vacc0xCDEF, vmultiplier);
    vacc1x0123 = vqdmulhq_s32(vacc1x0123, vmultiplier);
    vacc1x4567 = vqdmulhq_s32(vacc1x4567, vmultiplier);
    vacc1x89AB = vqdmulhq_s32(vacc1x89AB, vmultiplier);
    vacc1xCDEF = vqdmulhq_s32(vacc1xCDEF, vmultiplier);
    vacc2x0123 = vqdmulhq_s32(vacc2x0123, vmultiplier);
    vacc2x4567 = vqdmulhq_s32(vacc2x4567, vmultiplier);
    vacc2x89AB = vqdmulhq_s32(vacc2x89AB, vmultiplier);
    vacc2xCDEF = vqdmulhq_s32(vacc2xCDEF, vmultiplier);
    vacc3x0123 = vqdmulhq_s32(vacc3x0123, vmultiplier);
    vacc3x4567 = vqdmulhq_s32(vacc3x4567, vmultiplier);
    vacc3x89AB = vqdmulhq_s32(vacc3x89AB, vmultiplier);
    vacc3xCDEF = vqdmulhq_s32(vacc3xCDEF, vmultiplier);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_post_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_post_shift);
    vacc0x89AB = vrshlq_s32(vacc0x89AB, vright_post_shift);
    vacc0xCDEF = vrshlq_s32(vacc0xCDEF, vright_post_shift);
    vacc1x0123 = vrshlq_s32(vacc1x0123, vright_post_shift);
    vacc1x4567 = vrshlq_s32(vacc1x4567, vright_post_shift);
    vacc1x89AB = vrshlq_s32(vacc1x89AB, vright_post_shift);
    vacc1xCDEF = vrshlq_s32(vacc1xCDEF, vright_post_shift);
    vacc2x0123 = vrshlq_s32(vacc2x0123, vright_post_shift);
    vacc2x4567 = vrshlq_s32(vacc2x4567, vright_post_shift);
    vacc2x89AB = vrshlq_s32(vacc2x89AB, vright_post_shift);
    vacc2xCDEF = vrshlq_s32(vacc2xCDEF, vright_post_shift);
    vacc3x0123 = vrshlq_s32(vacc3x0123, vright_post_shift);
    vacc3x4567 = vrshlq_s32(vacc3x4567, vright_post_shift);
    vacc3x89AB = vrshlq_s32(vacc3x89AB, vright_post_shift);
    vacc3xCDEF = vrshlq_s32(vacc3xCDEF, vright_post_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->rndnu_neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
    const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x89AB), vacc1xCDEF), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
    const int16x8_t vacc2x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x89AB), vacc2xCDEF), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);
    const int16x8_t vacc3x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x89AB), vacc3xCDEF), voutput_zero_point);

    int8x16_t vout0x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc0x89ABCDEF);
    int8x16_t vout1x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc1x01234567), vacc1x89ABCDEF);
    int8x16_t vout2x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc2x01234567), vacc2x89ABCDEF);
    int8x16_t vout3x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc3x01234567), vacc3x89ABCDEF);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF)), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
    const int16x8_t vacc1x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x89AB), vqmovn_s32(vacc1xCDEF)), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
    const int16x8_t vacc2x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x89AB), vqmovn_s32(vacc2xCDEF)), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);
    const int16x8_t vacc3x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x89AB), vqmovn_s32(vacc3xCDEF)), voutput_zero_point);

    int8x16_t vout0x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc0x89ABCDEF));
    int8x16_t vout1x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc1x01234567), vqmovn_s16(vacc1x89ABCDEF));
    int8x16_t vout2x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc2x01234567), vqmovn_s16(vacc2x89ABCDEF));
    int8x16_t vout3x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc3x01234567), vqmovn_s16(vacc3x89ABCDEF));
#endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->rndnu_neon.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->rndnu_neon.output_max);

    vout0x0123456789ABCDEF = vmaxq_s8(vout0x0123456789ABCDEF, voutput_min);
    vout1x0123456789ABCDEF = vmaxq_s8(vout1x0123456789ABCDEF, voutput_min);
    vout2x0123456789ABCDEF = vmaxq_s8(vout2x0123456789ABCDEF, voutput_min);
    vout3x0123456789ABCDEF = vmaxq_s8(vout3x0123456789ABCDEF, voutput_min);

    vout0x0123456789ABCDEF = vminq_s8(vout0x0123456789ABCDEF, voutput_max);
    vout1x0123456789ABCDEF = vminq_s8(vout1x0123456789ABCDEF, voutput_max);
    vout2x0123456789ABCDEF = vminq_s8(vout2x0123456789ABCDEF, voutput_max);
    vout3x0123456789ABCDEF = vminq_s8(vout3x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_s8(c3 + 0, vout3x0123456789ABCDEF);
      vst1q_s8(c2 + 0, vout2x0123456789ABCDEF);
      vst1q_s8(c1 + 0, vout1x0123456789ABCDEF);
      vst1q_s8(c0 + 0, vout0x0123456789ABCDEF);

      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      int8x16_t vout2x01234567_3x01234567 = vcombine_s8(vget_low_s8(vout2x0123456789ABCDEF), vget_low_s8(vout3x0123456789ABCDEF));
      int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vget_low_s8(vout0x0123456789ABCDEF), vget_low_s8(vout1x0123456789ABCDEF));
      if (nc & 8) {
        vst1_s8(c3, vget_high_s8(vout2x01234567_3x01234567)); c3 += 8;
        vst1_s8(c2, vget_low_s8(vout2x01234567_3x01234567)); c2 += 8;
        vst1_s8(c1, vget_high_s8(vout0x01234567_1x01234567)); c1 += 8;
        vst1_s8(c0, vget_low_s8(vout0x01234567_1x01234567)); c0 += 8;
        vout2x01234567_3x01234567 = vcombine_s8(vget_high_s8(vout2x0123456789ABCDEF), vget_high_s8(vout3x0123456789ABCDEF));
        vout0x01234567_1x01234567 = vcombine_s8(vget_high_s8(vout0x0123456789ABCDEF), vget_high_s8(vout1x0123456789ABCDEF));
      }
      if (nc & 4) {
        vst1q_lane_u32(__builtin_assume_aligned(c3, 1), vreinterpretq_u32_s8(vout2x01234567_3x01234567), 2); c3 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c2, 1), vreinterpretq_u32_s8(vout2x01234567_3x01234567), 0); c2 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c1, 1), vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16(__builtin_assume_aligned(c3, 1), vreinterpretq_u16_s8(vout2x01234567_3x01234567), 4); c3 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c2, 1), vreinterpretq_u16_s8(vout2x01234567_3x01234567), 0); c2 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c1, 1), vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c0, 1), vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c3, vout2x01234567_3x01234567, 8);
        vst1q_lane_s8(c2, vout2x01234567_3x01234567, 0);
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
