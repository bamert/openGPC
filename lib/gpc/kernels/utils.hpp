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
#ifndef __NDB__KERNEL_UTILS
#define __NDB__KERNEL_UTILS

#include <cassert>
#include <thread>

#include "gpc/buffer.hpp"
using namespace std;

#ifdef _INTRINSICS_SSE
#include <immintrin.h>
// greater and lesser than simd ops for unsigned 8bit integer (epu8)
#define _mm_cmpgt_epu8(v0, v1)                             \
    _mm_cmpgt_epi8(_mm_xor_si128(v0, _mm_set1_epi8(-128)), \
                   _mm_xor_si128(v1, _mm_set1_epi8(-128)))
#define _mm_cmplt_epu8(v1, v0)                             \
    _mm_cmpgt_epi8(_mm_xor_si128(v0, _mm_set1_epi8(-128)), \
                   _mm_xor_si128(v1, _mm_set1_epi8(-128)))
#endif
namespace ndb {
/**
 * @brief Gets indices of non-zero values in array  a.
 *    Credits:
 *    https://stackoverflow.com/questions/18971401/sparse-array-compression-using-simd-avx2/41958528#41958528
 *
 * @param     input array
 * @param n   number of input elements
 * @param ind output array (indices into n of nonzero elements)
 * @param m   number of elements in output
 */
void arr2ind(const unsigned char* a,
                                       int n,
                                       int* ind,
                                       int* m);

#ifdef _INTRINSICS_SSE
/**
 * @brief      Unpacks 16x8bit from a 128bit simd var into 2x128bit vars
 *             (8x16bit)
 *
 * @param[in]  x     the 128 bit vector to be unpacked
 * @param      y0    The y 0
 * @param      y1    The y 1
 */
void unpack8to16(const __m128i x, __m128i& y0, __m128i& y1);

/**
 * @brief      Packs 2x128bit vars with 16bit values(where 8 upper bits are
 *             zero) into 1x128bit with 8bit values
 *
 * @param[in]  x0    The x 0
 * @param[in]  x1    The x 1
 * @param      y     the packed vector
 */
void pack16to8(const __m128i x0, const __m128i x1, __m128i& y);
#endif
/**
 * @brief Calls a given functional f with subranges based on the given start
 *        and end indices. Here the functional is assumed to take two integer
 *        arguments indicating their respective start and end ranges.
 *        nThreads determines the number of threads the given range shall be
 * split into. The range is inclusive on the lower bound and exclusive on the
 * upper bound, i.e. [start,end)
 *
 * @param f        function object (e.g. a lambda functional)
 * @param start    start of the range
 * @param end      end of the range
 * @param nThreads number of threads to use
 */
void parFor(std::function<void(int, int)> const& f,
            int start,
            int end,
            int nThreads);

}  // namespace ndb
#endif
