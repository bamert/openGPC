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
namespace ndb {
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
            int sx = (*p11 + *p31 + 2 * *p21 - *p13 - 2 * *p23 - *p33) / 9;
            int sy = (*p11 + *p13 + 2 * *p12 - *p31 - 2 * *p32 - *p33) / 9;

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
void sobel(uint8_t* in,
           uint8_t* blurred,
           int width,
           int height,
           uint8_t threshold,
           int numThreads) {
    assert(width % 16 == 0 && "width must be multiple of 16!");
#ifndef _INTRINSICS_SSE
    sobelNaive(in, blurred, width, height, threshold);
#else
    auto sobelSSESegment = [&](int start, int end) {
        __m128i one_third, one_ninth, one, two, mone, mtwo, binThres;
        __m128i *dst0, *dst1;
        __m128i zero = _mm_setzero_si128();

        int x, y;
        one_third = _mm_set1_epi16(
            21846);  // 2^16/3+1. For 16bit ints. 2^8/3+1=86.33 for 8bit
        one_ninth = _mm_set1_epi16(7282);  // 2^16/9+1. For 16bit ints.

        binThres = _mm_set1_epi16(threshold * threshold);

        dst0 = (__m128i*)(blurred + width * 1);
        // dst1 = (__m128i *)(blurred + width * 2);
        for (y = start; y < end;
             y++) {  // We compute results for two rows in one iteration
            const uint8_t *row0, *row1, *row2;

            row1 = in + y * width;
            row0 = row1 - width;
            row2 = row1 + width;

            for (x = 0; x < width; x += 16) {
                // Note: Center element not used in sobel kernels!!
                // Kernel indices:
                // 00 01 02
                // 10 11 12
                // 20 21 22

                __m128i a00, a01, a02, a10, a12, a20, a21, a22;
                __m128i b00, b01, b02, b10, b12, b20, b21, b22;

                __m128i raA, raB, rbA, rbB;
                __m128i tmpa, tmpb, sya, syb, sxa, sxb, res;

                unpack8to16(_mm_loadu_si128((__m128i*)(row0 - 1)), a00, b00);
                unpack8to16(_mm_load_si128((__m128i*)(row0)), a01, b01);
                unpack8to16(_mm_loadu_si128((__m128i*)(row0 + 1)), a02, b02);

                unpack8to16(_mm_loadu_si128((__m128i*)(row1 - 1)), a10, b10);
                unpack8to16(_mm_loadu_si128((__m128i*)(row1 + 1)), a12, b12);

                unpack8to16(_mm_loadu_si128((__m128i*)(row2 - 1)), a20, b20);
                unpack8to16(_mm_load_si128((__m128i*)(row2)), a21, b21);
                unpack8to16(_mm_loadu_si128((__m128i*)(row2 + 1)), a22, b22);

                // Sobel kernels for x and y direction.
                //      1 0 -1       1 2 1
                // sx = 2 0 -2 sy =  0 0 0
                //      1 0 -1      -1-2-1
                //      Note that neither kernel uses the center element)

                // In the following, mullo is used to multiply intermediate
                // results with -1 To divide by 3, 16bit overflow divide by
                // multiply is used, which thus uses the upper 16bit(_mm_mulhi)
                // of the 32bit temporary result.

                // sx column kernel vectors (1,2,1)
                // Two chained add/sub are used for 2 and -2
                raA = _mm_mulhi_epi16(
                    _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(a00, a20), a10),
                                  a10),
                    one_ninth);
                rbA = _mm_mulhi_epi16(
                    _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(b00, b20), b10),
                                  b10),
                    one_ninth);

                // sx column kernel vector (-1 -2 -1)
                raB = _mm_mulhi_epi16(
                    _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(a02, a22), a12),
                                  a12),
                    one_ninth);
                rbB = _mm_mulhi_epi16(
                    _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(b02, b22), b12),
                                  b12),
                    one_ninth);

                // Square of sx: Add squares of above temporaries into final sum
                tmpa = _mm_sub_epi16(raA, raB);
                tmpb = _mm_sub_epi16(rbA, rbB);

                sxa = _mm_mullo_epi16(tmpa, tmpa);
                sxb = _mm_mullo_epi16(tmpb, tmpb);

                // sy row kernel vector (1,2,1)
                // Two chained add are used for 2 and -2
                raA = _mm_mulhi_epi16(
                    _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(a00, a02), a01),
                                  a01),
                    one_ninth);
                rbA = _mm_mulhi_epi16(
                    _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(b00, b02), b01),
                                  b01),
                    one_ninth);

                // sy row kernel vector (-1 -2 -1)
                raB = _mm_mulhi_epi16(
                    _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(a20, a22), a21),
                                  a21),
                    one_ninth);
                rbB = _mm_mulhi_epi16(
                    _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(b20, b22), b21),
                                  b21),
                    one_ninth);

                // Square of sx: Add squares of above temporaries into final sum
                tmpa = _mm_sub_epi16(raA, raB);
                tmpb = _mm_sub_epi16(rbA, rbB);

                // watch out, can't overwrite this
                sya = _mm_mullo_epi16(tmpa, tmpa);
                syb = _mm_mullo_epi16(tmpb, tmpb);

                __m128i zero = _mm_setzero_si128();

                // The unpacklo is necessary because _mm_cmput_epi16 sets the
                // output to 0xFFFF if the comparison is true. When packing
                // 16bit to 8bit however, 0xFFFF will be interpreted (in a
                // signed environment) as being negative, and hence set to 0,
                // resulting in a 0 output everywhere. using unpacklo in between
                // we get 0xFFFF->0xFF
                pack16to8(
                    _mm_unpacklo_epi8(
                        _mm_cmpgt_epi16(_mm_adds_epi16(sxa, sya), binThres),
                        zero),
                    _mm_unpacklo_epi8(
                        _mm_cmpgt_epi16(_mm_adds_epi16(sxb, syb), binThres),
                        zero),
                    res);

                _mm_store_si128(dst0++, res);

                row0 += 16;
                row1 += 16;
                row2 += 16;
            }  // cols
        }  // rows
    };  // Lambda
    sobelSSESegment(1, height - 3);
#endif
}
} // namespace ndb
