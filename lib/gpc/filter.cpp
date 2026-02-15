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
#ifndef __NDB__FILTER
#define __NDB__FILTER

#include <cassert>
#include <thread>

#include "gpc/filter.hpp"
using namespace std;

namespace ndb {
void arr2ind(const unsigned char* a,
                                       int n,
                                       int* ind,
                                       int* m) {
#ifdef _INTRINSICS_SSE
    int i, m0, k;
    __m256i msk;
    m0 = 0;
    for (i = 0; i < n; i = i + 32) { /* Load 32 bytes and compare with zero: */
        msk = _mm256_cmpeq_epi8(_mm256_load_si256((__m256i*)&a[i]),
                                _mm256_setzero_si256());
        k = _mm256_movemask_epi8(msk);
        k = ~k; /* Search for nonzero bits instead of zero bits.  */
        while (k) {
            ind[m0] =
                i + _tzcnt_u32(
                        k); /* Count the number of trailing zero bits in k. */
            m0++;
            k = _blsr_u32(k); /* Clear the lowest set bit in k. */
        }
    }
    *m = m0;
#else
    int nnz = 0;
    for (int i = 0; i < n; i++) {
        if (a[i] != 0) {
            nnz++;
            *ind = i;
            ind++;
        }
    }
    *m = nnz;
#endif
}
#ifdef _INTRINSICS_SSE
void unpack8to16(const __m128i x, __m128i& y0, __m128i& y1) {
    __m128i zero = _mm_setzero_si128();
    y0 = _mm_unpacklo_epi8(x, zero);
    y1 = _mm_unpackhi_epi8(x, zero);
}
void pack16to8(const __m128i x0, const __m128i x1, __m128i& y) {
    y = _mm_packus_epi16(x0, x1);
}

#endif
void parFor(std::function<void(int, int)> const& f,
            int start,
            int end,
            int nThreads) {
    // Range definition
    // quantities derived from range
    int segSize = (end - start) / nThreads;
    int lastSeg = (end - start) % nThreads;

    std::vector<std::thread> threads;
    threads.reserve(nThreads);

    // Spawn threads
    for (int t = 0; t < nThreads - 1; t++) {
        threads.emplace_back(f, start + t * segSize, start + (t + 1) * segSize);
    }
    threads.emplace_back(f,
                         start + (nThreads - 1) * segSize,
                         start + (nThreads)*segSize + lastSeg);
    // Join
    for (auto& t : threads) t.join();
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
void box(uint8_t* in, uint8_t* blurred, int width, int height, int numThreads) {
    assert(width % 16 == 0 && "width must be multiple of 16!");
#ifndef _INTRINSICS_SSE
    boxNaive(in, blurred, width, height);
#else
    auto boxFilterSegment = [&](int start, int end) {
        int x, y;
        __m128i one_third;
        __m128i *dst0, *dst1;
        __m128i zero = _mm_setzero_si128();

        one_third = _mm_set1_epi16(
            21846);  // 2^16/3+1. For 16bit ints. 2^8/3+1=86.33 for 8bit
        dst0 = (__m128i*)(blurred + width * (start));
        dst1 = (__m128i*)(blurred + width * (start + 1));
        for (y = start; y < end;
             y += 2) {  // We compute results for two rows in one iteration
            const uint8_t *row0, *row1, *row2, *row3;

            row1 = in + y * width;
            row0 = row1 - width;
            row2 = row1 + width;
            row3 = row2 + width;

            for (x = 0; x < width; x += 16) {
                __m128i s00, s01, s02;
                __m128i r00, r01, r02;
                __m128i ra00, ra01, ra02;
                __m128i rb00, rb01, rb02;

                __m128i a00, a01, a02, b00, b01, b02;

                __m128i tmp0, tmp1, res;

                s00 = _mm_loadu_si128((__m128i*)(row0 - 1));
                s01 = _mm_loadu_si128((__m128i*)(row0 + 1));
                s02 = _mm_load_si128((__m128i*)(row0));
                unpack8to16(s00, a00, b00);
                unpack8to16(s01, a01, b01);
                unpack8to16(s02, a02, b02);

                ra00 = _mm_mulhi_epi16(
                    _mm_adds_epi16(_mm_adds_epi16(a00, a01), a02), one_third);
                rb00 = _mm_mulhi_epi16(
                    _mm_adds_epi16(_mm_adds_epi16(b00, b01), b02), one_third);

                s00 = _mm_loadu_si128((__m128i*)(row1 - 1));
                s01 = _mm_loadu_si128((__m128i*)(row1 + 1));
                s02 = _mm_load_si128((__m128i*)(row1));
                unpack8to16(s00, a00, b00);
                unpack8to16(s01, a01, b01);
                unpack8to16(s02, a02, b02);

                ra01 = _mm_mulhi_epi16(
                    _mm_adds_epi16(_mm_adds_epi16(a00, a01), a02), one_third);
                rb01 = _mm_mulhi_epi16(
                    _mm_adds_epi16(_mm_adds_epi16(b00, b01), b02), one_third);

                s00 = _mm_loadu_si128((__m128i*)(row2 - 1));
                s01 = _mm_loadu_si128((__m128i*)(row2 + 1));
                s02 = _mm_load_si128((__m128i*)(row2));
                unpack8to16(s00, a00, b00);
                unpack8to16(s01, a01, b01);
                unpack8to16(s02, a02, b02);

                ra02 = _mm_mulhi_epi16(
                    _mm_adds_epi16(_mm_adds_epi16(a00, a01), a02), one_third);
                rb02 = _mm_mulhi_epi16(
                    _mm_adds_epi16(_mm_adds_epi16(b00, b01), b02), one_third);

                tmp0 = _mm_mulhi_epi16(
                    _mm_adds_epi16(_mm_adds_epi16(ra00, ra01), ra02),
                    one_third);
                tmp1 = _mm_mulhi_epi16(
                    _mm_adds_epi16(_mm_adds_epi16(rb00, rb01), rb02),
                    one_third);

                pack16to8(tmp0, tmp1, res);
                _mm_store_si128(dst0++, res);

                s00 = _mm_loadu_si128((__m128i*)(row3 - 1));
                s01 = _mm_loadu_si128((__m128i*)(row3 + 1));
                s02 = _mm_load_si128((__m128i*)(row3));
                unpack8to16(s00, a00, b00);
                unpack8to16(s01, a01, b01);
                unpack8to16(s02, a02, b02);
                ra00 = _mm_mulhi_epi16(
                    _mm_adds_epi16(_mm_adds_epi16(a00, a01), a02), one_third);
                rb00 = _mm_mulhi_epi16(
                    _mm_adds_epi16(_mm_adds_epi16(b00, b01), b02), one_third);

                tmp0 = _mm_mulhi_epi16(
                    _mm_adds_epi16(_mm_adds_epi16(ra00, ra01), ra02),
                    one_third);
                tmp1 = _mm_mulhi_epi16(
                    _mm_adds_epi16(_mm_adds_epi16(rb00, rb01), rb02),
                    one_third);

                pack16to8(tmp0, tmp1, res);
                _mm_store_si128(dst1++, res);

                row0 += 16;
                row1 += 16;
                row2 += 16;
                row3 += 16;
            }
            // still storing 128bit, but now in 16 x 8bit format, so /16 instead
            // of /8
            dst0 += width / 16;
            dst1 += width / 16;
        }
    };  // lambda

    boxFilterSegment(1, height - 3);
    // parFor(boxFilterSegment,1,height-3,4);
#endif
}


#ifdef _INTRINSICS_SSE
bool isAllZeros(__m128i xmm) {
    return _mm_movemask_epi8(_mm_cmpeq_epi8(xmm, _mm_setzero_si128())) ==
           0xFFFF;
}
#endif
void gpcFilter(uint8_t* in,
               const uint8_t* grad,
               uint32_t* gpc,
               std::vector<int32_t> fastmask,
               std::vector<int>& idx,
               int width,
               int height,
               int numThreads) {
    assert(width % 16 == 0 && "width must be multiple of 16!");
#ifndef _INTRINSICS_SSE
    gpcFilterNaive(in, grad, gpc, fastmask, idx, width, height);
#else
    auto gpcFilterSegment = [&](int start, int end) {
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
                                _mm_lddqu_si128(
                                    (__m128i*)(center + fastmask[i])),
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
    };

    if (numThreads == 1)
        gpcFilterSegment(13, height - 15);
    else
        parFor(gpcFilterSegment, 13, height - 15, 4);
#endif
}
void gpcFilterTau(uint8_t* in,
                  const uint8_t* grad,
                  uint32_t* gpc,
                  std::vector<int32_t> fastmask,
                  std::vector<int> tau,
                  std::vector<int>& idx,
                  int width,
                  int height,
                  int numThreads) {
    assert(width % 16 == 0 && "width must be multiple of 16!");
#ifndef _INTRINSICS_SSE
    gpcFilterTauNaive(in, grad, gpc, fastmask, tau, idx, width, height);
#else
    auto gpcFilterSegment = [&](int start, int end) {
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
                                _mm_lddqu_si128(
                                    (__m128i*)(center + fastmask[i])),
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
    };

    if (numThreads == 1)
        gpcFilterSegment(13, height - 15);
    else
        parFor(gpcFilterSegment, 13, height - 15, 4);
#endif
}
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
}  // namespace ndb
#endif
