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
#include <cassert>
#include "gpc/kernels/census.hpp"
void census5x5Naive(uint8_t* in, uint32_t* census, int width, int height) {
    uint32_t val;
    uint32_t* dst;
    for (int y = 2; y < height - 3; y++) {
        for (int x = 0; x < width; x++) {
            val = 0;
            dst = census + y * width + x;
            int i = 0;
            // patch loops
            for (int px = -2; px <= 2; px++) {
                for (int py = -2; py <= 2; py++) {
                    if (!(px == 0 && py == 0)) {
                        val |= (in[(y + py) * width + (x + px)] >
                                in[y * width + x])
                                   ? (1 << i)
                                   : 0;
                        i++;
                    }
                }
            }  // End patch loops
            *dst = val;
        }
    }  // End pixel loops
}
void census5x5(uint8_t* in, uint32_t* census, int width, int height) {
    assert(width % 16 == 0 && "width must be multiple of 16!");
#ifndef _INTRINSICS_SSE
    census5x5Naive(in, census, width, height);
#else
    __m128i zero = _mm_set1_epi8(0);
    __m128i one = _mm_set1_epi8(1);

    for (int y = 2; y < height - 3; y++) {
        for (int x = 0; x < width; x += 16) {
            uint8_t* rowPtr;
            rowPtr = in + (y - 2) * width + x;
            __m128i center = _mm_lddqu_si128((__m128i*)(in + y * width + x));
            __m128i* dst = (__m128i*)(census + y * width +
                                      x);  // Set starting point to pixel (2,2)
            // row 0
            __m128i bitMask = one;
            __m128i byte1 = _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr - 2))),
                bitMask);
            bitMask += bitMask;  // 2
            byte1 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr - 1))),
                bitMask);
            bitMask += bitMask;  // 4
            byte1 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr))),
                bitMask);
            bitMask += bitMask;  // 8
            byte1 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr + 1))),
                bitMask);
            bitMask += bitMask;  // 16
            byte1 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr + 2))),
                bitMask);

            // row 1
            rowPtr += width;
            bitMask += bitMask;  // 32
            byte1 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr - 2))),
                bitMask);
            bitMask += bitMask;  // 64
            byte1 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr - 1))),
                bitMask);
            bitMask += bitMask;  // 128
            byte1 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr))),
                bitMask);
            bitMask = one;  // 1
            __m128i byte2 = _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr + 1))),
                bitMask);
            bitMask += bitMask;  // 2
            byte2 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr + 2))),
                bitMask);

            // row 2
            rowPtr += width;
            bitMask += bitMask;  // 4
            byte2 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr - 2))),
                bitMask);
            bitMask += bitMask;  // 8
            byte2 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr - 1))),
                bitMask);
            bitMask += bitMask;  // 16
            byte2 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr + 1))),
                bitMask);
            bitMask += bitMask;  // 32
            byte2 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr + 2))),
                bitMask);

            // row 3
            rowPtr += width;
            bitMask += bitMask;  // 64
            byte2 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr - 2))),
                bitMask);
            bitMask += bitMask;  // 128
            byte2 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr - 1))),
                bitMask);
            bitMask = one;  // 1
            __m128i byte3 = _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr))),
                bitMask);
            bitMask += bitMask;  // 2
            byte3 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr + 1))),
                bitMask);
            bitMask += bitMask;  // 4
            byte3 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr + 2))),
                bitMask);

            // row 4
            rowPtr += width;
            bitMask += bitMask;  // 8
            byte3 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr - 2))),
                bitMask);
            bitMask += bitMask;  // 16
            byte3 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr - 1))),
                bitMask);
            bitMask += bitMask;  // 32
            byte3 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr))),
                bitMask);
            bitMask += bitMask;  // 64
            byte3 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr + 1))),
                bitMask);
            bitMask += bitMask;  // 128
            byte3 |= _mm_and_si128(
                _mm_cmplt_epu8(center, _mm_lddqu_si128((__m128i*)(rowPtr + 2))),
                bitMask);

            // 8bit to 16bit
            __m128i high1 = _mm_unpacklo_epi8(byte3, zero);
            __m128i high2 = _mm_unpackhi_epi8(byte3, zero);
            __m128i low1 = _mm_unpacklo_epi8(byte1, byte2);
            __m128i low2 = _mm_unpackhi_epi8(byte1, byte2);

            // 16bit to 32bit ints
            _mm_storeu_si128(dst, _mm_unpacklo_epi16(low1, high1));
            _mm_storeu_si128(dst + 1, _mm_unpackhi_epi16(low1, high1));
            _mm_storeu_si128(dst + 2, _mm_unpacklo_epi16(low2, high2));
            _mm_storeu_si128(dst + 3, _mm_unpackhi_epi16(low2, high2));

        }  // col iteration
    }  // row iteration
    // if(numThreads == 1)
    // gpcFilterSegment(13,height-15);
    // else
    // parFor(gpcFilterSegment,13,height-15,4);

#endif
}  // census5x5


