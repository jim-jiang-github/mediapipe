// Auto-generated file. Do not edit!
//   Template: src/f32-vrnd/scalar-libm.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vunary.h>


void xnn_f32_vrndu_ukernel__scalar_libm_x1(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  do {
    const float vx = *x++;
    const float vy = ceilf(vx);
    *y++ = vy;
    n -= sizeof(float);
  } while (n != 0);
}
