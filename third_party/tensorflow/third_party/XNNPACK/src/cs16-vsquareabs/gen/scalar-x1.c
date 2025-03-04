// Auto-generated file. Do not edit!
//   Template: src/cs16-vsquareabs/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack/math.h>
#include <xnnpack/vsquareabs.h>


void xnn_cs16_vsquareabs_ukernel__scalar_x1(
    size_t batch,
    const int16_t* input,
    uint32_t* output) {

  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);


 if XNN_UNLIKELY(batch != 0) {
   do {
     const int32_t vr = (int32_t) input[0];
     const int32_t vi = (int32_t) input[1];
     input += 2;

     const uint32_t vrsquare = (uint32_t) (vr * vr);
     const uint32_t visquare = (uint32_t) (vi * vi);

     const uint32_t vout = vrsquare + visquare;

     *output++ = vout;
   } while (--batch != 0);
 }
}
