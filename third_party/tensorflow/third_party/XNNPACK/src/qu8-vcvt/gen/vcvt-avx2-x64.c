// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/avx2.c.in
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
#include <xnnpack/vcvt.h>


void xnn_qu8_vcvt_ukernel__avx2_x64(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union xnn_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint8_t) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const __m256i vinput_zero_point = _mm256_load_si256((const __m256i*) params->avx2.input_zero_point);
  const __m256i vmultiplier = _mm256_load_si256((const __m256i*) params->avx2.multiplier);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx2.output_zero_point);
  for (; n >= 64 * sizeof(uint8_t); n -= 64 * sizeof(uint8_t)) {
    __m256i vacc0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) x));
    __m256i vacc1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) (x + 16)));
    __m256i vacc2 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) (x + 32)));
    __m256i vacc3 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) (x + 48)));
    x += 64;

    vacc0 = _mm256_sub_epi16(vinput_zero_point, vacc0);
    vacc1 = _mm256_sub_epi16(vinput_zero_point, vacc1);
    vacc2 = _mm256_sub_epi16(vinput_zero_point, vacc2);
    vacc3 = _mm256_sub_epi16(vinput_zero_point, vacc3);

    vacc0 = _mm256_slli_epi16(vacc0, 7);
    vacc1 = _mm256_slli_epi16(vacc1, 7);
    vacc2 = _mm256_slli_epi16(vacc2, 7);
    vacc3 = _mm256_slli_epi16(vacc3, 7);

    vacc0 = _mm256_mulhrs_epi16(vacc0, vmultiplier);
    vacc1 = _mm256_mulhrs_epi16(vacc1, vmultiplier);
    vacc2 = _mm256_mulhrs_epi16(vacc2, vmultiplier);
    vacc3 = _mm256_mulhrs_epi16(vacc3, vmultiplier);

    vacc0 = _mm256_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm256_adds_epi16(vacc1, voutput_zero_point);
    vacc2 = _mm256_adds_epi16(vacc2, voutput_zero_point);
    vacc3 = _mm256_adds_epi16(vacc3, voutput_zero_point);

    __m256i vy0 = _mm256_packus_epi16(vacc0, vacc1);
    __m256i vy1 = _mm256_packus_epi16(vacc2, vacc3);

    vy0 = _mm256_permute4x64_epi64(vy0, _MM_SHUFFLE(3, 1, 2, 0));
    vy1 = _mm256_permute4x64_epi64(vy1, _MM_SHUFFLE(3, 1, 2, 0));

    _mm256_storeu_si256((__m256i*) y, vy0);
    _mm256_storeu_si256((__m256i*) (y + 32), vy1);
    y += 64;
  }
  for (; n >= 16 * sizeof(uint8_t); n -= 16 * sizeof(uint8_t)) {
    __m256i vacc = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) x));
    vacc = _mm256_sub_epi16(vinput_zero_point, vacc);
    vacc = _mm256_slli_epi16(vacc, 7);
    vacc = _mm256_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm256_adds_epi16(vacc, voutput_zero_point);
    x += 16;

    const __m128i vacc_hi = _mm256_extracti128_si256(vacc, 1);
    const __m128i vy = _mm_packus_epi16(_mm256_castsi256_si128(vacc), vacc_hi);
    _mm_storeu_si128((__m128i*) y, vy);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(uint8_t));
    assert(n <= 15 * sizeof(uint8_t));

    __m256i vacc = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*) x));
    vacc = _mm256_sub_epi16(vinput_zero_point, vacc);
    vacc = _mm256_slli_epi16(vacc, 7);
    vacc = _mm256_mulhrs_epi16(vacc, vmultiplier);
    vacc = _mm256_adds_epi16(vacc, voutput_zero_point);

    const __m128i vacc_hi = _mm256_extracti128_si256(vacc, 1);
    __m128i vy = _mm_packus_epi16(_mm256_castsi256_si128(vacc), vacc_hi);
    if (n & (8 * sizeof(uint8_t))) {
      _mm_storel_epi64((__m128i*) y, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      y += 8;
    }
    if (n & (4 * sizeof(uint8_t))) {
      _mm_storeu_si32(y, vy);
      vy = _mm_srli_epi64(vy, 32);
      y += 4;
    }
    if (n & (2 * sizeof(uint8_t))) {
      _mm_storeu_si16(y, vy);
      vy = _mm_srli_epi32(vy, 16);
      y += 2;
    }
    if (n & (1 * sizeof(uint8_t))) {
      *y = (uint8_t) _mm_extract_epi8(vy, 0);
    }
  }
}
