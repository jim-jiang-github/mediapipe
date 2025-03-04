#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/qs8-vcvt/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-neon-x8.c &
tools/xngen src/qs8-vcvt/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-neon-x16.c &
tools/xngen src/qs8-vcvt/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-neon-x32.c &

tools/xngen src/qs8-vcvt/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-neon-x8.c &
tools/xngen src/qs8-vcvt/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-neon-x16.c &
tools/xngen src/qs8-vcvt/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-neon-x32.c &

################################### x86 SSE2 ##################################
tools/xngen src/qs8-vcvt/sse2.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-sse2-x16.c &
tools/xngen src/qs8-vcvt/sse2.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-sse2-x32.c &

tools/xngen src/qs8-vcvt/sse2.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-sse2-x16.c &
tools/xngen src/qs8-vcvt/sse2.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-sse2-x32.c &

################################## x86 SSSE3 ##################################
tools/xngen src/qs8-vcvt/ssse3.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-ssse3-x16.c &
tools/xngen src/qs8-vcvt/ssse3.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-ssse3-x32.c &

tools/xngen src/qs8-vcvt/ssse3.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-ssse3-x16.c &
tools/xngen src/qs8-vcvt/ssse3.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-ssse3-x32.c &

################################## x86 SSE4.1 #################################
tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=8  -D AVX=0 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-sse41-x8.c &
tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=16 -D AVX=0 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-sse41-x16.c &
tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=32 -D AVX=0 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-sse41-x32.c &

tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=8  -D AVX=1 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-avx-x8.c &
tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=16 -D AVX=1 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-avx-x16.c &
tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=32 -D AVX=1 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-avx-x32.c &

tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=8  -D AVX=0 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-sse41-x8.c &
tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=16 -D AVX=0 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-sse41-x16.c &
tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=32 -D AVX=0 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-sse41-x32.c &

tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=8  -D AVX=1 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-avx-x8.c &
tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=16 -D AVX=1 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-avx-x16.c &
tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=32 -D AVX=1 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-avx-x32.c &

################################### x86 AVX2 ##################################
tools/xngen src/qs8-vcvt/avx2.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-avx2-x16.c &
tools/xngen src/qs8-vcvt/avx2.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-avx2-x32.c &
tools/xngen src/qs8-vcvt/avx2.c.in -D BATCH_TILE=64 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-avx2-x64.c &

tools/xngen src/qs8-vcvt/avx2.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-avx2-x16.c &
tools/xngen src/qs8-vcvt/avx2.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-avx2-x32.c &
tools/xngen src/qs8-vcvt/avx2.c.in -D BATCH_TILE=64 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-avx2-x64.c &

################################## WAsm SIMD ##################################
tools/xngen src/qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-wasmsimd-x8.c &
tools/xngen src/qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-wasmsimd-x16.c &
tools/xngen src/qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-wasmsimd-x32.c &

tools/xngen src/qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-wasmsimd-x8.c &
tools/xngen src/qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-wasmsimd-x16.c &
tools/xngen src/qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-wasmsimd-x32.c &

################################## ARMv6 SIMD #################################
tools/xngen src/qs8-vcvt/armv6simd.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-armv6simd-x4.c &
tools/xngen src/qs8-vcvt/armv6simd.c.in -D BATCH_TILE=8 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-armv6simd-x8.c &

tools/xngen src/qs8-vcvt/armv6simd.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-armv6simd-x4.c &
tools/xngen src/qs8-vcvt/armv6simd.c.in -D BATCH_TILE=8 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-armv6simd-x8.c &

#################################### Scalar ###################################
tools/xngen src/qs8-vcvt/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-scalar-x1.c &
tools/xngen src/qs8-vcvt/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-scalar-x2.c &
tools/xngen src/qs8-vcvt/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/vcvt-scalar-x4.c &

tools/xngen src/qs8-vcvt/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-scalar-x1.c &
tools/xngen src/qs8-vcvt/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-scalar-x2.c &
tools/xngen src/qs8-vcvt/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/vcvt-scalar-x4.c &

################################## Unit tests #################################
tools/generate-vcvt-test.py --spec test/qs8-vcvt.yaml --output test/qs8-vcvt.cc &
tools/generate-vcvt-test.py --spec test/qu8-vcvt.yaml --output test/qu8-vcvt.cc &

wait
