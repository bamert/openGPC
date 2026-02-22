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
#include <thread>
#include <functional>
#include "gpc/kernels/utils.hpp"

using namespace std;

namespace ndb {
void arr2ind(const unsigned char* a,
                                       int n,
                                       int* ind,
                                       int* m) {
#if HWY_TARGET == HWY_AVX2
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
#if HWY_TARGET == HWY_AVX2
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





}  // namespace ndb
