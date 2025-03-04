// Auto-generated file. Do not edit!
//   Template: src/f16-vsigmoid/avx2.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vunary.h>


void xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x48(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch % sizeof(uint16_t) == 0);

  const __m256 vsign_mask = _mm256_load_ps(params->avx2_rr1_p2.sign_mask);
  const __m256 vmagic_bias = _mm256_load_ps(params->avx2_rr1_p2.magic_bias);
  const __m256 vlog2e = _mm256_load_ps(params->avx2_rr1_p2.log2e);
  const __m256 vminus_ln2 = _mm256_load_ps(params->avx2_rr1_p2.minus_ln2);
  const __m256 vc2 = _mm256_load_ps(params->avx2_rr1_p2.c2);
  const __m256 vc1 = _mm256_load_ps(params->avx2_rr1_p2.c1);
  const __m256 vone = _mm256_load_ps(params->avx2_rr1_p2.one);
  const __m256 vdenorm_cutoff = _mm256_load_ps(params->avx2_rr1_p2.denorm_cutoff);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 48 * sizeof(uint16_t); batch -= 48 * sizeof(uint16_t)) {
    const __m256 vx0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    const __m256 vx1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    const __m256 vx2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 16)));
    const __m256 vx3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 24)));
    const __m256 vx4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 32)));
    const __m256 vx5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 40)));
    i += 48;

    const __m256 vz0 = _mm256_or_ps(vx0, vsign_mask);
    const __m256 vz1 = _mm256_or_ps(vx1, vsign_mask);
    const __m256 vz2 = _mm256_or_ps(vx2, vsign_mask);
    const __m256 vz3 = _mm256_or_ps(vx3, vsign_mask);
    const __m256 vz4 = _mm256_or_ps(vx4, vsign_mask);
    const __m256 vz5 = _mm256_or_ps(vx5, vsign_mask);

    __m256 vn0 = _mm256_fmadd_ps(vz0, vlog2e, vmagic_bias);
    __m256 vn1 = _mm256_fmadd_ps(vz1, vlog2e, vmagic_bias);
    __m256 vn2 = _mm256_fmadd_ps(vz2, vlog2e, vmagic_bias);
    __m256 vn3 = _mm256_fmadd_ps(vz3, vlog2e, vmagic_bias);
    __m256 vn4 = _mm256_fmadd_ps(vz4, vlog2e, vmagic_bias);
    __m256 vn5 = _mm256_fmadd_ps(vz5, vlog2e, vmagic_bias);

    const __m256 vs0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn0), 23));
    const __m256 vs1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn1), 23));
    const __m256 vs2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn2), 23));
    const __m256 vs3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn3), 23));
    const __m256 vs4 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn4), 23));
    const __m256 vs5 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn5), 23));

    vn0 = _mm256_sub_ps(vn0, vmagic_bias);
    vn1 = _mm256_sub_ps(vn1, vmagic_bias);
    vn2 = _mm256_sub_ps(vn2, vmagic_bias);
    vn3 = _mm256_sub_ps(vn3, vmagic_bias);
    vn4 = _mm256_sub_ps(vn4, vmagic_bias);
    vn5 = _mm256_sub_ps(vn5, vmagic_bias);

    __m256 vt0 = _mm256_fmadd_ps(vn0, vminus_ln2, vz0);
    __m256 vt1 = _mm256_fmadd_ps(vn1, vminus_ln2, vz1);
    __m256 vt2 = _mm256_fmadd_ps(vn2, vminus_ln2, vz2);
    __m256 vt3 = _mm256_fmadd_ps(vn3, vminus_ln2, vz3);
    __m256 vt4 = _mm256_fmadd_ps(vn4, vminus_ln2, vz4);
    __m256 vt5 = _mm256_fmadd_ps(vn5, vminus_ln2, vz5);

    const __m256 vp0 = _mm256_fmadd_ps(vc2, vt0, vc1);
    const __m256 vp1 = _mm256_fmadd_ps(vc2, vt1, vc1);
    const __m256 vp2 = _mm256_fmadd_ps(vc2, vt2, vc1);
    const __m256 vp3 = _mm256_fmadd_ps(vc2, vt3, vc1);
    const __m256 vp4 = _mm256_fmadd_ps(vc2, vt4, vc1);
    const __m256 vp5 = _mm256_fmadd_ps(vc2, vt5, vc1);

    vt0 = _mm256_mul_ps(vt0, vs0);
    vt1 = _mm256_mul_ps(vt1, vs1);
    vt2 = _mm256_mul_ps(vt2, vs2);
    vt3 = _mm256_mul_ps(vt3, vs3);
    vt4 = _mm256_mul_ps(vt4, vs4);
    vt5 = _mm256_mul_ps(vt5, vs5);

    const __m256 ve0 = _mm256_fmadd_ps(vt0, vp0, vs0);
    const __m256 ve1 = _mm256_fmadd_ps(vt1, vp1, vs1);
    const __m256 ve2 = _mm256_fmadd_ps(vt2, vp2, vs2);
    const __m256 ve3 = _mm256_fmadd_ps(vt3, vp3, vs3);
    const __m256 ve4 = _mm256_fmadd_ps(vt4, vp4, vs4);
    const __m256 ve5 = _mm256_fmadd_ps(vt5, vp5, vs5);

    const __m256 vd0 = _mm256_add_ps(ve0, vone);
    const __m256 vd1 = _mm256_add_ps(ve1, vone);
    const __m256 vd2 = _mm256_add_ps(ve2, vone);
    const __m256 vd3 = _mm256_add_ps(ve3, vone);
    const __m256 vd4 = _mm256_add_ps(ve4, vone);
    const __m256 vd5 = _mm256_add_ps(ve5, vone);

    __m256 vf0 = _mm256_div_ps(ve0, vd0);
    __m256 vf1 = _mm256_div_ps(ve1, vd1);
    __m256 vf2 = _mm256_div_ps(ve2, vd2);
    __m256 vf3 = _mm256_div_ps(ve3, vd3);
    __m256 vf4 = _mm256_div_ps(ve4, vd4);
    __m256 vf5 = _mm256_div_ps(ve5, vd5);

    vf0 = _mm256_andnot_ps(_mm256_cmp_ps(vz0, vdenorm_cutoff, _CMP_LT_OS), vf0);
    vf1 = _mm256_andnot_ps(_mm256_cmp_ps(vz1, vdenorm_cutoff, _CMP_LT_OS), vf1);
    vf2 = _mm256_andnot_ps(_mm256_cmp_ps(vz2, vdenorm_cutoff, _CMP_LT_OS), vf2);
    vf3 = _mm256_andnot_ps(_mm256_cmp_ps(vz3, vdenorm_cutoff, _CMP_LT_OS), vf3);
    vf4 = _mm256_andnot_ps(_mm256_cmp_ps(vz4, vdenorm_cutoff, _CMP_LT_OS), vf4);
    vf5 = _mm256_andnot_ps(_mm256_cmp_ps(vz5, vdenorm_cutoff, _CMP_LT_OS), vf5);

    vf0 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf0), vf0, vx0);
    vf1 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf1), vf1, vx1);
    vf2 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf2), vf2, vx2);
    vf3 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf3), vf3, vx3);
    vf4 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf4), vf4, vx4);
    vf5 = _mm256_blendv_ps(_mm256_sub_ps(vone, vf5), vf5, vx5);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vf0, _MM_FROUND_NO_EXC));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vf1, _MM_FROUND_NO_EXC));
    _mm_storeu_si128((__m128i*) (o + 16), _mm256_cvtps_ph(vf2, _MM_FROUND_NO_EXC));
    _mm_storeu_si128((__m128i*) (o + 24), _mm256_cvtps_ph(vf3, _MM_FROUND_NO_EXC));
    _mm_storeu_si128((__m128i*) (o + 32), _mm256_cvtps_ph(vf4, _MM_FROUND_NO_EXC));
    _mm_storeu_si128((__m128i*) (o + 40), _mm256_cvtps_ph(vf5, _MM_FROUND_NO_EXC));
    o += 48;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    const __m256 vz = _mm256_or_ps(vx, vsign_mask);

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));
    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    const __m256 vp = _mm256_fmadd_ps(vc2, vt, vc1);
    vt = _mm256_mul_ps(vt, vs);
    const __m256 ve = _mm256_fmadd_ps(vt, vp, vs);

    const __m256 vd = _mm256_add_ps(ve, vone);
    __m256 vf = _mm256_div_ps(ve, vd);

    vf = _mm256_andnot_ps(_mm256_cmp_ps(vz, vdenorm_cutoff, _CMP_LT_OS), vf);
    vf = _mm256_blendv_ps(_mm256_sub_ps(vone, vf), vf, vx);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vf, _MM_FROUND_NO_EXC));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    const __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));

    const __m256 vz = _mm256_or_ps(vx, vsign_mask);

    __m256 vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);
    const __m256 vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));
    vn = _mm256_sub_ps(vn, vmagic_bias);

    __m256 vt = _mm256_fmadd_ps(vn, vminus_ln2, vz);

    const __m256 vp = _mm256_fmadd_ps(vc2, vt, vc1);
    vt = _mm256_mul_ps(vt, vs);
    const __m256 ve = _mm256_fmadd_ps(vt, vp, vs);

    const __m256 vd = _mm256_add_ps(ve, vone);
    __m256 vf = _mm256_div_ps(ve, vd);

    vf = _mm256_andnot_ps(_mm256_cmp_ps(vz, vdenorm_cutoff, _CMP_LT_OS), vf);
    vf = _mm256_blendv_ps(_mm256_sub_ps(vone, vf), vf, vx);

    __m128i vh = _mm256_cvtps_ph(vf, _MM_FROUND_NO_EXC);
    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      o += 4;
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vh);
      vh = _mm_srli_epi64(vh, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}
