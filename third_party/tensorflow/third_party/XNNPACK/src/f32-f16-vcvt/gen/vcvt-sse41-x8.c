// Auto-generated file. Do not edit!
//   Template: src/f32-f16-vcvt/sse.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <smmintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/unaligned.h>
#include <xnnpack/vcvt.h>


void xnn_f32_f16_vcvt_ukernel__sse41_x8(
    size_t n,
    const float* input,
    void* output,
    const union xnn_f32_f16_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vnonsign_mask = _mm_load_ps((const float*) params->sse2.nonsign_mask);
  const __m128i vexp_bias = _mm_load_si128((const __m128i*) params->sse2.exp_bias);
  const __m128 vscale_to_inf = _mm_load_ps(params->sse2.scale_to_inf);
  const __m128i vexpw_max = _mm_load_si128((const __m128i*) params->sse2.expw_max);
  const __m128 vscale_to_zero = _mm_load_ps(params->sse2.scale_to_zero);
  const __m128i vbias_min = _mm_load_si128((const __m128i*) params->sse2.bias_min);
  const __m128i vmanth_mask = _mm_load_si128((const __m128i*) params->sse2.manth_mask);
  const __m128i vexph_mask = _mm_load_si128((const __m128i*) params->sse2.exph_mask);
  const __m128i vnanh = _mm_load_si128((const __m128i*) params->sse2.nanh);

  uint16_t* o = (uint16_t*) output;
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const __m128 vx_lo = _mm_loadu_ps(input);
    const __m128 vx_hi = _mm_loadu_ps(input + 4);
    input += 8;

    const __m128 vabsx_lo = _mm_and_ps(vx_lo, vnonsign_mask);
    const __m128 vabsx_hi = _mm_and_ps(vx_hi, vnonsign_mask);

    const __m128 vsignx_lo = _mm_xor_ps(vx_lo, vabsx_lo);
    const __m128 vsignx_hi = _mm_xor_ps(vx_hi, vabsx_hi);
    __m128i vbias_lo = _mm_add_epi32(_mm_castps_si128(vabsx_lo), vexp_bias);
    __m128i vbias_hi = _mm_add_epi32(_mm_castps_si128(vabsx_hi), vexp_bias);
    __m128 vf_lo = _mm_mul_ps(vabsx_lo, vscale_to_inf);
    __m128 vf_hi = _mm_mul_ps(vabsx_hi, vscale_to_inf);
    const __m128i vnanmaskw_lo = _mm_cmpgt_epi32(_mm_castps_si128(vabsx_lo), vexpw_max);
    const __m128i vnanmaskw_hi = _mm_cmpgt_epi32(_mm_castps_si128(vabsx_hi), vexpw_max);

    vbias_lo = _mm_and_si128(vbias_lo, vexpw_max);
    vbias_hi = _mm_and_si128(vbias_hi, vexpw_max);
    vf_lo = _mm_mul_ps(vf_lo, vscale_to_zero);
    vf_hi = _mm_mul_ps(vf_hi, vscale_to_zero);
    const __m128i vnanmaskh = _mm_packs_epi32(vnanmaskw_lo, vnanmaskw_hi);
    const __m128i vsignh = _mm_packs_epi32(_mm_castps_si128(vsignx_lo), _mm_castps_si128(vsignx_hi));

    vbias_lo = _mm_max_epi16(vbias_lo, vbias_min);
    vbias_hi = _mm_max_epi16(vbias_hi, vbias_min);

    vf_lo = _mm_add_ps(vf_lo, _mm_castsi128_ps(vbias_lo));
    vf_hi = _mm_add_ps(vf_hi, _mm_castsi128_ps(vbias_hi));

    __m128i vexpw_lo = _mm_srli_epi32(_mm_castps_si128(vf_lo), 13);
    __m128i vexpw_hi = _mm_srli_epi32(_mm_castps_si128(vf_hi), 13);
    const __m128i vmantw_lo = _mm_and_si128(_mm_castps_si128(vf_lo), vmanth_mask);
    const __m128i vmantw_hi = _mm_and_si128(_mm_castps_si128(vf_hi), vmanth_mask);

    vexpw_lo = _mm_and_si128(vexpw_lo, vexph_mask);
    vexpw_hi = _mm_and_si128(vexpw_hi, vexph_mask);

    const __m128i vnonsignw_lo = _mm_add_epi32(vmantw_lo, vexpw_lo);
    const __m128i vnonsignw_hi = _mm_add_epi32(vmantw_hi, vexpw_hi);

    const __m128i vnonsignh = _mm_packs_epi32(vnonsignw_lo, vnonsignw_hi);

    const __m128i vabsh = _mm_blendv_epi8(vnonsignh, vnanh, vnanmaskh);

    const __m128i vh = _mm_or_si128(vabsh, vsignh);

    _mm_storeu_si128((__m128i*) o, vh);
    o += 8;
  }
  if XNN_UNPREDICTABLE(n != 0) {
    const __m128 vx_lo = _mm_loadu_ps(input);
    const float* input_hi = (const float*) ((uintptr_t) input + (n & (4 * sizeof(float))));
    const __m128 vx_hi = _mm_loadu_ps(input_hi);

    const __m128 vabsx_lo = _mm_and_ps(vx_lo, vnonsign_mask);
    const __m128 vabsx_hi = _mm_and_ps(vx_hi, vnonsign_mask);

    const __m128 vsignx_lo = _mm_xor_ps(vx_lo, vabsx_lo);
    const __m128 vsignx_hi = _mm_xor_ps(vx_hi, vabsx_hi);
    __m128i vbias_lo = _mm_add_epi32(_mm_castps_si128(vabsx_lo), vexp_bias);
    __m128i vbias_hi = _mm_add_epi32(_mm_castps_si128(vabsx_hi), vexp_bias);
    __m128 vf_lo = _mm_mul_ps(vabsx_lo, vscale_to_inf);
    __m128 vf_hi = _mm_mul_ps(vabsx_hi, vscale_to_inf);
    const __m128i vnanmaskw_lo = _mm_cmpgt_epi32(_mm_castps_si128(vabsx_lo), vexpw_max);
    const __m128i vnanmaskw_hi = _mm_cmpgt_epi32(_mm_castps_si128(vabsx_hi), vexpw_max);

    vbias_lo = _mm_and_si128(vbias_lo, vexpw_max);
    vbias_hi = _mm_and_si128(vbias_hi, vexpw_max);
    vf_lo = _mm_mul_ps(vf_lo, vscale_to_zero);
    vf_hi = _mm_mul_ps(vf_hi, vscale_to_zero);
    const __m128i vnanmaskh = _mm_packs_epi32(vnanmaskw_lo, vnanmaskw_hi);
    const __m128i vsignh = _mm_packs_epi32(_mm_castps_si128(vsignx_lo), _mm_castps_si128(vsignx_hi));

    vbias_lo = _mm_max_epi16(vbias_lo, vbias_min);
    vbias_hi = _mm_max_epi16(vbias_hi, vbias_min);

    vf_lo = _mm_add_ps(vf_lo, _mm_castsi128_ps(vbias_lo));
    vf_hi = _mm_add_ps(vf_hi, _mm_castsi128_ps(vbias_hi));

    __m128i vexpw_lo = _mm_srli_epi32(_mm_castps_si128(vf_lo), 13);
    __m128i vexpw_hi = _mm_srli_epi32(_mm_castps_si128(vf_hi), 13);
    const __m128i vmantw_lo = _mm_and_si128(_mm_castps_si128(vf_lo), vmanth_mask);
    const __m128i vmantw_hi = _mm_and_si128(_mm_castps_si128(vf_hi), vmanth_mask);

    vexpw_lo = _mm_and_si128(vexpw_lo, vexph_mask);
    vexpw_hi = _mm_and_si128(vexpw_hi, vexph_mask);

    const __m128i vnonsignw_lo = _mm_add_epi32(vmantw_lo, vexpw_lo);
    const __m128i vnonsignw_hi = _mm_add_epi32(vmantw_hi, vexpw_hi);

    const __m128i vnonsignh = _mm_packs_epi32(vnonsignw_lo, vnonsignw_hi);

    const __m128i vabsh = _mm_blendv_epi8(vnonsignh, vnanh, vnanmaskh);

    __m128i vh = _mm_or_si128(vabsh, vsignh);

    if (n & (4 * sizeof(float))) {
      _mm_storel_epi64((__m128i*) o, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      o += 4;
    }
    if (n & (2 * sizeof(float))) {
      unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(vh));
      vh = _mm_srli_epi64(vh, 32);
      o += 2;
    }
    if (n & (1 * sizeof(float))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}
