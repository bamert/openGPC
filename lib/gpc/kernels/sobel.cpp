// Copyright (c) 2018, ETH Zurich
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Code Author: Niklaus Bamert (bamertn@ethz.ch)
#include "gpc/kernels/sobel.hpp"

#include <cassert>

#include "gpc/kernels/utils.hpp"
namespace ndb {
namespace testing {
void sobel_hwy(
    uint8_t* in, uint8_t* blurred, int width, int height, uint8_t threshold);
}
void sobelNaive(
    uint8_t* in, uint8_t* gradient, int width, int height, uint8_t threshold) {
    assert(width % 16 == 0 && "width must be multiple of 16!");
    int thresholdSq = threshold * threshold;
    uint8_t* ptr = in;

    uint8_t* p11 = ptr + 0 * width;
    uint8_t* p12 = ptr + 0 * width + 1;
    uint8_t* p13 = ptr + 0 * width + 2;

    uint8_t* p21 = ptr + 1 * width;
    uint8_t* p22 = ptr + 1 * width + 1;
    uint8_t* p23 = ptr + 1 * width + 2;

    uint8_t* p31 = ptr + 2 * width;
    uint8_t* p32 = ptr + 2 * width + 1;
    uint8_t* p33 = ptr + 2 * width + 2;

    // output pointer
    uint8_t* optr = gradient + 1 * width + 1;
    // Apply 3x3 box filter to image less pixel border of 1 (to avoid treating
    // boundary) (unoptimized)
    for (int iy = 1; iy < height - 1; iy++) {
        for (int ix = 0; ix < width; ix++) {
            // Approximate division by 9 with fixed-point multiplication (2^16/9
            // = 7282)
            int16_t sum_x = (*p11 + *p31 + 2 * *p21 - *p13 - 2 * *p23 - *p33);
            int16_t sum_y = (*p11 + *p13 + 2 * *p12 - *p31 - 2 * *p32 - *p33);

            int sx = (static_cast<int32_t>(sum_x) * 7282) >> 16;
            int sy = (static_cast<int32_t>(sum_y) * 7282) >> 16;
            int val = sx * sx + sy * sy;

            *optr = val > thresholdSq ? 255 : 0;
            p11++;
            p12++;
            p13++;
            p21++;
            p22++;
            p23++;
            p31++;
            p32++;
            p33++;
            optr++;
        }
    }
}
// #ifdef _INTRINSICS_SSE
#if HWY_TARGET == HWY_AVX2
#include <immintrin.h>

void sobelSSE(const uint8_t* in,
              uint8_t* blurred,
              int width,
              int start,
              int end,
              uint8_t threshold) {
    __m128i zero = _mm_setzero_si128();
    __m128i one_ninth = _mm_set1_epi16(7282);  // 2^16/9
    __m128i binThres = _mm_set1_epi16(threshold * threshold);

    for (int y = start; y < end; y++) {
        const uint8_t* row1 = in + y * width;
        const uint8_t* row0 = row1 - width;
        const uint8_t* row2 = row1 + width;

        // Output destination for this specific row
        __m128i* dst = (__m128i*)(blurred + y * width + 1);

        for (int x = 0; x < width; x += 16) {
            __m128i a00, a01, a02, a10, a12, a20, a21, a22;
            __m128i b00, b01, b02, b10, b12, b20, b21, b22;
            __m128i raA, raB, rbA, rbB;
            __m128i tmpa, tmpb, sya, syb, sxa, sxb, res;

            // Load and unpack 3x3 neighborhood (excluding center a11/b11)
            unpack8to16(_mm_loadu_si128((__m128i*)(row0 + x - 1)), a00, b00);
            unpack8to16(_mm_loadu_si128((__m128i*)(row0 + x)), a01, b01);
            unpack8to16(_mm_loadu_si128((__m128i*)(row0 + x + 1)), a02, b02);

            unpack8to16(_mm_loadu_si128((__m128i*)(row1 + x - 1)), a10, b10);
            unpack8to16(_mm_loadu_si128((__m128i*)(row1 + x + 1)), a12, b12);

            unpack8to16(_mm_loadu_si128((__m128i*)(row2 + x - 1)), a20, b20);
            unpack8to16(_mm_loadu_si128((__m128i*)(row2 + x)), a21, b21);
            unpack8to16(_mm_loadu_si128((__m128i*)(row2 + x + 1)), a22, b22);

            // --- SX Calculation ---
            // Left col (1,2,1)
            raA = _mm_mulhi_epi16(
                _mm_add_epi16(_mm_add_epi16(a00, a20), _mm_add_epi16(a10, a10)),
                one_ninth);
            rbA = _mm_mulhi_epi16(
                _mm_add_epi16(_mm_add_epi16(b00, b20), _mm_add_epi16(b10, b10)),
                one_ninth);
            // Right col (-1,-2,-1)
            raB = _mm_mulhi_epi16(
                _mm_add_epi16(_mm_add_epi16(a02, a22), _mm_add_epi16(a12, a12)),
                one_ninth);
            rbB = _mm_mulhi_epi16(
                _mm_add_epi16(_mm_add_epi16(b02, b22), _mm_add_epi16(b12, b12)),
                one_ninth);

            tmpa = _mm_sub_epi16(raA, raB);
            tmpb = _mm_sub_epi16(rbA, rbB);
            sxa = _mm_mullo_epi16(tmpa, tmpa);
            sxb = _mm_mullo_epi16(tmpb, tmpb);

            // --- SY Calculation ---
            // Top row (1,2,1)
            raA = _mm_mulhi_epi16(
                _mm_add_epi16(_mm_add_epi16(a00, a02), _mm_add_epi16(a01, a01)),
                one_ninth);
            rbA = _mm_mulhi_epi16(
                _mm_add_epi16(_mm_add_epi16(b00, b02), _mm_add_epi16(b01, b01)),
                one_ninth);
            // Bottom row (-1,-2,-1)
            raB = _mm_mulhi_epi16(
                _mm_add_epi16(_mm_add_epi16(a20, a22), _mm_add_epi16(a21, a21)),
                one_ninth);
            rbB = _mm_mulhi_epi16(
                _mm_add_epi16(_mm_add_epi16(b20, b22), _mm_add_epi16(b21, b21)),
                one_ninth);

            tmpa = _mm_sub_epi16(raA, raB);
            tmpb = _mm_sub_epi16(rbA, rbB);
            sya = _mm_mullo_epi16(tmpa, tmpa);
            syb = _mm_mullo_epi16(tmpb, tmpb);

            // --- Thresholding and Packing ---
            pack16to8(
                _mm_unpacklo_epi8(
                    _mm_cmpgt_epi16(_mm_adds_epi16(sxa, sya), binThres), zero),
                _mm_unpacklo_epi8(
                    _mm_cmpgt_epi16(_mm_adds_epi16(sxb, syb), binThres), zero),
                res);

            _mm_storeu_si128(dst++, res);
        }
    }
}
#endif
void sobel(uint8_t* in,
           uint8_t* blurred,
           int width,
           int height,
           uint8_t threshold,
           int numThreads) {
    assert(width % 16 == 0 && "width must be multiple of 16!");
#if defined(__ARM_NEON) || defined(__aarch64__)
    sobelNaive(in, blurred, width, height, threshold);
    // testing::sobel_hwy(in, blurred, width, height, threshold); // not exact!
#else
#ifndef _INTRINSICS_SSE
    sobelNaive(in, blurred, width, height, threshold);
#else
    sobelSSE(in, blurred, width, 1, height - 1, threshold);
#endif
#endif
}
}  // namespace ndb
