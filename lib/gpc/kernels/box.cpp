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

#include "gpc/kernels/box.hpp"
#include "gpc/kernels/utils.hpp"
#include <cassert>
namespace ndb {
namespace testing { 
    void box_hwy(uint8_t* in, uint8_t* blurred, int width, int height); 
}
void boxNaive(uint8_t* in, uint8_t* blurred, int width, int height) {
    assert(width % 16 == 0 && "width must be multiple of 16!");
    // allocate space for result
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
    uint8_t* optr = blurred + 1 * width + 1;

    // Apply 3x3 box filter to image less pixel border of 1 (to avoid treating
    // boundary) (unoptimized)
    for (int iy = 1; iy < height - 1; iy++) {
        for (int ix = 0; ix < width; ix++) {
            int res =
                (*p11 + *p12 + *p13 + *p21 + *p22 + *p23 + *p31 + *p32 + *p33) /
                9;
            *optr = res;
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
#if HWY_TARGET == HWY_AVX2
/**
 * @brief SSE implementation of the 3x3 box filter.
 * Processed two rows at a time using fixed-point multiplication for division.
 */
#include <immintrin.h>
void boxSSE(uint8_t* in, uint8_t* blurred, int width, int height) {
    int start = 1;
    int end = height - 3;
    
    int x, y;
    __m128i one_third = _mm_set1_epi16(21846); // 2^16/3 + 1
    
    __m128i *dst0 = (__m128i*)(blurred + width * start);
    __m128i *dst1 = (__m128i*)(blurred + width * (start + 1));

    for (y = start; y < end; y += 2) {
        const uint8_t *row0, *row1, *row2, *row3;

        row1 = in + y * width;
        row0 = row1 - width;
        row2 = row1 + width;
        row3 = row2 + width;

        for (x = 0; x < width; x += 16) {
            __m128i s00, s01, s02;
            __m128i ra00, ra01, ra02, rb00, rb01, rb02;
            __m128i a00, a01, a02, b00, b01, b02;
            __m128i tmp0, tmp1, res;

            // Row 0 Processing
            s00 = _mm_loadu_si128((__m128i*)(row0 - 1));
            s01 = _mm_loadu_si128((__m128i*)(row0 + 1));
            s02 = _mm_load_si128((__m128i*)(row0));
            unpack8to16(s00, a00, b00);
            unpack8to16(s01, a01, b01);
            unpack8to16(s02, a02, b02);
            ra00 = _mm_mulhi_epi16(_mm_adds_epi16(_mm_adds_epi16(a00, a01), a02), one_third);
            rb00 = _mm_mulhi_epi16(_mm_adds_epi16(_mm_adds_epi16(b00, b01), b02), one_third);

            // Row 1 Processing
            s00 = _mm_loadu_si128((__m128i*)(row1 - 1));
            s01 = _mm_loadu_si128((__m128i*)(row1 + 1));
            s02 = _mm_load_si128((__m128i*)(row1));
            unpack8to16(s00, a00, b00);
            unpack8to16(s01, a01, b01);
            unpack8to16(s02, a02, b02);
            ra01 = _mm_mulhi_epi16(_mm_adds_epi16(_mm_adds_epi16(a00, a01), a02), one_third);
            rb01 = _mm_mulhi_epi16(_mm_adds_epi16(_mm_adds_epi16(b00, b01), b02), one_third);

            // Row 2 Processing
            s00 = _mm_loadu_si128((__m128i*)(row2 - 1));
            s01 = _mm_loadu_si128((__m128i*)(row2 + 1));
            s02 = _mm_load_si128((__m128i*)(row2));
            unpack8to16(s00, a00, b00);
            unpack8to16(s01, a01, b01);
            unpack8to16(s02, a02, b02);
            ra02 = _mm_mulhi_epi16(_mm_adds_epi16(_mm_adds_epi16(a00, a01), a02), one_third);
            rb02 = _mm_mulhi_epi16(_mm_adds_epi16(_mm_adds_epi16(b00, b01), b02), one_third);

            // Accumulate rows 0, 1, 2 for dst0
            tmp0 = _mm_mulhi_epi16(_mm_adds_epi16(_mm_adds_epi16(ra00, ra01), ra02), one_third);
            tmp1 = _mm_mulhi_epi16(_mm_adds_epi16(_mm_adds_epi16(rb00, rb01), rb02), one_third);
            pack16to8(tmp0, tmp1, res);
            _mm_store_si128(dst0++, res);

            // Row 3 Processing
            s00 = _mm_loadu_si128((__m128i*)(row3 - 1));
            s01 = _mm_loadu_si128((__m128i*)(row3 + 1));
            s02 = _mm_load_si128((__m128i*)(row3));
            unpack8to16(s00, a00, b00);
            unpack8to16(s01, a01, b01);
            unpack8to16(s02, a02, b02);
            ra00 = _mm_mulhi_epi16(_mm_adds_epi16(_mm_adds_epi16(a00, a01), a02), one_third);
            rb00 = _mm_mulhi_epi16(_mm_adds_epi16(_mm_adds_epi16(b00, b01), b02), one_third);

            // Accumulate rows 1, 2, 3 for dst1
            tmp0 = _mm_mulhi_epi16(_mm_adds_epi16(_mm_adds_epi16(ra01, ra02), ra00), one_third);
            tmp1 = _mm_mulhi_epi16(_mm_adds_epi16(_mm_adds_epi16(rb01, rb02), rb00), one_third);
            pack16to8(tmp0, tmp1, res);
            _mm_store_si128(dst1++, res);

            row0 += 16; row1 += 16; row2 += 16; row3 += 16;
        }
        dst0 += width / 16;
        dst1 += width / 16;
    }
}
#endif
void box(uint8_t* in, uint8_t* blurred, int width, int height, int numThreads) {
    assert(width % 16 == 0 && "width must be multiple of 16!");
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Force use of our new Highway kernel on Mac
    testing::box_hwy(in, blurred, width, height);
#else
    #if HWY_TARGET == HWY_AVX2
        boxSSE(in, blurred, width, height);
    #else
        boxNaive(in, blurred, width, height);
    #endif
#endif
}
}  // namespace ndb
