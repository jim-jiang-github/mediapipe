// Auto-generated file. Do not edit!
//   Template: src/f16-f32-vcvt/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vcvt.h>


void xnn_f16_f32_vcvt_ukernel__scalar_x2(
    size_t n,
    const void* input,
    float* output,
    const union xnn_f16_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint32_t vsign_mask = params->scalar.sign_mask;
  const uint32_t vexp_offset = params->scalar.exp_offset;
  const float vexp_scale = params->scalar.exp_scale;
  const uint32_t vmagic_mask = params->scalar.magic_mask;
  const float vmagic_bias = params->scalar.magic_bias;
  const uint32_t vdenorm_cutoff = params->scalar.denorm_cutoff;

  const uint16_t* i = (const uint16_t*) input;
  uint32_t* o = (uint32_t*) output;
  for (; n >= 2 * sizeof(uint16_t); n -= 2 * sizeof(uint16_t)) {
    const uint16_t vh0 = i[0];
    const uint16_t vh1 = i[1];
    i += 2;

    const uint32_t vw0 = (uint32_t) vh0 << 16;
    const uint32_t vw1 = (uint32_t) vh1 << 16;

    const uint32_t vsign0 = vw0 & vsign_mask;
    const uint32_t vsign1 = vw1 & vsign_mask;

    const uint32_t v2w0 = vw0 + vw0;
    const uint32_t v2w1 = vw1 + vw1;

    const uint32_t vnorm0 = float_as_uint32(uint32_as_float((v2w0 >> 4) + vexp_offset) * vexp_scale);
    const uint32_t vnorm1 = float_as_uint32(uint32_as_float((v2w1 >> 4) + vexp_offset) * vexp_scale);

    const uint32_t vdenorm0 = float_as_uint32(uint32_as_float((v2w0 >> 17) | vmagic_mask) - vmagic_bias);
    const uint32_t vdenorm1 = float_as_uint32(uint32_as_float((v2w1 >> 17) | vmagic_mask) - vmagic_bias);

    const uint32_t vf0 = vsign0 | (XNN_UNPREDICTABLE(v2w0 < vdenorm_cutoff) ? vdenorm0 : vnorm0);
    const uint32_t vf1 = vsign1 | (XNN_UNPREDICTABLE(v2w1 < vdenorm_cutoff) ? vdenorm1 : vnorm1);

    o[0] = vf0;
    o[1] = vf1;
    o += 2;
  }
  if XNN_UNLIKELY(n != 0) {
    const uint16_t vh = *i;

    const uint32_t vw = (uint32_t) vh << 16;
    const uint32_t vsign = vw & vsign_mask;
    const uint32_t v2w = vw + vw;
    const uint32_t vnorm = float_as_uint32(uint32_as_float((v2w >> 4) + vexp_offset) * vexp_scale);
    const uint32_t vdenorm = float_as_uint32(uint32_as_float((v2w >> 17) | vmagic_mask) - vmagic_bias);
    const uint32_t vf = vsign | (XNN_UNPREDICTABLE(v2w < vdenorm_cutoff) ? vdenorm : vnorm);

    *o = vf;
  }
}
