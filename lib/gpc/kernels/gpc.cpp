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
#include "gpc/kernels/gpc.hpp"

#include <cassert>
namespace ndb {
void gpcFilterNaive(uint8_t* in,
                    const uint8_t* grad,
                    uint32_t* gpc,
                    std::vector<int32_t> fastmask,
                    std::vector<int>& idx,
                    int width,
                    int height) {
    // output buffer of same size
    uint32_t tmp;

    int j = 0;
    for (auto k : idx) {
        tmp = 0;
        for (uint8_t i = 0; i < fastmask.size(); i += 2) {
            tmp <<= 1;  // shift by one
            if (*(in + k + fastmask[i]) > *(in + k + fastmask[i + 1]))
                tmp++;  // set this test's result to 1
        }
        gpc[k] = tmp;
        j++;
    }
}

void gpcFilterTauNaive(uint8_t* in,
                       const uint8_t* grad,
                       uint32_t* gpc,
                       std::vector<int32_t> fastmask,
                       std::vector<int> tau,
                       std::vector<int>& idx,
                       int width,
                       int height) {
    uint32_t tmp;

    int j = 0;
    for (auto k : idx) {
        tmp = 0;
        for (uint8_t i = 0; i < fastmask.size(); i += 2) {
            tmp <<= 1;  // shift by one
            if (*(in + k + fastmask[i]) >
                *(in + k + fastmask[i + 1]) - tau[i / 2])
                tmp++;  // set this test's result to 1
        }
        gpc[k] = tmp;
        j++;
    }
}

#if (HWY_ARCH_X86) && (HWY_TARGET == HWY_AVX2)
bool isAllZeros(__m128i xmm) {
    return _mm_movemask_epi8(_mm_cmpeq_epi8(xmm, _mm_setzero_si128())) ==
           0xFFFF;
}
void gpcFilterSSE(uint8_t* in,
                  const uint8_t* grad,
                  uint32_t* gpc,
                  std::vector<int32_t> fastmask,
                  std::vector<int>& idx,
                  int width,
                  int height) {
    const int start = 13;
    const int end = height - 15;
    __m128i zero = _mm_set1_epi8(0);
    __m128i one = _mm_set1_epi8(1);
    for (int y = start; y < end; y++) {
        for (int x = 0; x < width; x += 16) {
            uint8_t* rowPtr;
            rowPtr = in + (y - 2) * width + x;
            __m128i out[4];  // temporary output vector of 4 128bit words

            const uint8_t* center = (in + y * width + x);
            const uint8_t* centerGrad = (grad + y * width + x);
            // We only process the current segment if there are any non-zero
            // values (high gradient pixels)
            if (!isAllZeros(_mm_lddqu_si128((__m128i*)centerGrad))) {
                __m128i* dst =
                    (__m128i*)(gpc + y * width +
                               x);  // Set starting point to pixel (2,2)
                out[0] = zero;
                out[1] = zero;
                out[2] = zero;
                out[3] = zero;
                uint8_t k = 0;
                __m128i bitMask = one;
                for (uint8_t i = 0; i < fastmask.size() && i < 64; i += 2) {
                    out[k] |= _mm_and_si128(
                        _mm_cmpgt_epu8(
                            _mm_lddqu_si128((__m128i*)(center + fastmask[i])),
                            _mm_lddqu_si128(
                                (__m128i*)(center + fastmask[i + 1]))),
                        bitMask);
                    // Keeps index into output vector and updates bit mask
                    if (i % 16 == 0 && i != 0) {
                        bitMask = one;
                        k++;
                    } else {
                        bitMask += bitMask;
                    }
                }
                // 8bit to 16bit
                __m128i high1 = _mm_unpacklo_epi8(out[2], out[3]);
                __m128i high2 = _mm_unpackhi_epi8(out[2], out[3]);
                __m128i low1 = _mm_unpacklo_epi8(out[0], out[1]);
                __m128i low2 = _mm_unpackhi_epi8(out[0], out[1]);

                // 16bit to 32bit ints
                _mm_storeu_si128(dst, _mm_unpacklo_epi16(low1, high1));
                _mm_storeu_si128(dst + 1, _mm_unpackhi_epi16(low1, high1));
                _mm_storeu_si128(dst + 2, _mm_unpacklo_epi16(low2, high2));
                _mm_storeu_si128(dst + 3, _mm_unpackhi_epi16(low2, high2));
            }
        }  // col iteration
    }  // row iteration
}
#endif
void gpcFilter(uint8_t* in,
               const uint8_t* grad,
               uint32_t* gpc,
               std::vector<int32_t> fastmask,
               std::vector<int>& idx,
               int width,
               int height) {
    assert(width % 16 == 0 && "width must be multiple of 16!");
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Replace with call to highway
    gpcFilterNaive(in, grad, gpc, fastmask, idx, width, height);
#else
#if (HWY_ARCH_X86) && (HWY_TARGET == HWY_AVX2)
    gpcFilterSSE(in, grad, gpc, fastmask, idx, width, height);
#else
    gpcFilterNaive(in, grad, gpc, fastmask, idx, width, height);
#endif
#endif
}

#if (HWY_ARCH_X86) && (HWY_TARGET == HWY_AVX2)
void gpcFilterTauSSE(uint8_t* in,
                     const uint8_t* grad,
                     uint32_t* gpc,
                     std::vector<int32_t> fastmask,
                     std::vector<int> tau,
                     std::vector<int>& idx,
                     int width,
                     int height) {
    const int start = 13;
    const int end = height - 15;
    __m128i zero = _mm_set1_epi8(0);
    __m128i one = _mm_set1_epi8(1);
    for (int y = start; y < end; y++) {
        for (int x = 0; x < width; x += 16) {
            uint8_t* rowPtr;
            rowPtr = in + (y - 2) * width + x;
            __m128i out[4];  // temporary output vector of 4 128bit words

            const uint8_t* center = (in + y * width + x);
            const uint8_t* centerGrad = (grad + y * width + x);
            // We only process the current segment if there are any non-zero
            // values (high gradient pixels)
            if (!isAllZeros(_mm_lddqu_si128((__m128i*)centerGrad))) {
                __m128i* dst =
                    (__m128i*)(gpc + y * width +
                               x);  // Set starting point to pixel (2,2)
                out[0] = zero;
                out[1] = zero;
                out[2] = zero;
                out[3] = zero;
                uint8_t k = 0;
                __m128i bitMask = one;
                for (uint8_t i = 0; i < fastmask.size() && i < 64; i += 2) {
                    out[k] |= _mm_and_si128(
                        _mm_cmpgt_epu8(
                            _mm_lddqu_si128((__m128i*)(center + fastmask[i])),
                            _mm_subs_epi8(
                                _mm_lddqu_si128(
                                    (__m128i*)(center + fastmask[i + 1])),
                                _mm_set1_epi8(tau[i / 2]))  // deduct tau
                            ),
                        bitMask);
                    // Keeps index into output vector and updates bit mask
                    if (i % 16 == 0 && i != 0) {
                        bitMask = one;
                        k++;
                    } else {
                        bitMask += bitMask;
                    }
                }
                // 8bit to 16bit
                __m128i high1 = _mm_unpacklo_epi8(out[2], out[3]);
                __m128i high2 = _mm_unpackhi_epi8(out[2], out[3]);
                __m128i low1 = _mm_unpacklo_epi8(out[0], out[1]);
                __m128i low2 = _mm_unpackhi_epi8(out[0], out[1]);

                // 16bit to 32bit ints
                _mm_storeu_si128(dst, _mm_unpacklo_epi16(low1, high1));
                _mm_storeu_si128(dst + 1, _mm_unpackhi_epi16(low1, high1));
                _mm_storeu_si128(dst + 2, _mm_unpacklo_epi16(low2, high2));
                _mm_storeu_si128(dst + 3, _mm_unpackhi_epi16(low2, high2));
            }
        }  // col iteration
    }  // row iteration
}
#endif

void gpcFilterTau(uint8_t* in,
                  const uint8_t* grad,
                  uint32_t* gpc,
                  std::vector<int32_t> fastmask,
                  std::vector<int> tau,
                  std::vector<int>& idx,
                  int width,
                  int height) {
    assert(width % 16 == 0 && "width must be multiple of 16!");
#if defined(__ARM_NEON) || defined(__aarch64__)
    // Replace with call to highway
    gpcFilterTauNaive(in, grad, gpc, fastmask, tau, idx, width, height);
#else
#if (HWY_ARCH_X86) && (HWY_TARGET == HWY_AVX2)
    gpcFilterTauSSE(in, grad, gpc, fastmask, tau, idx, width, height);
#else
    gpcFilterTauNaive(in, grad, gpc, fastmask, tau, idx, width, height);
#endif
#endif
}
}  // namespace ndb
