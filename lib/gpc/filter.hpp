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
/**
 * @brief Naive 3x3 sobel filter implementation
 *
 * @param      in       input image
 * @param      blurred  The blurred output image
 * @param[in]  width    The width
 * @param[in]  height   The height
 * @param[in]  numThreads number of threads to use
 * @param      threshold  threshold to binarize sobel filter output
 */
void sobelNaive(
    uint8_t* in, uint8_t* gradient, int width, int height, uint8_t threshold);

/**
 * @brief      Naive 3x3 box filter implementation
 *
 * @param      in       input image
 * @param      blurred  The blurred output image
 * @param[in]  width    The width
 * @param[in]  height   The height
 * @param[in]  numThreads number of threads to use
 */
void boxNaive(uint8_t* in, uint8_t* blurred, int width, int height);

/**
 * @brief Applies a gpc filter defined by the pixel-difference tests in
 * fastmask. Naive implementation
 *
 * @param in        The input image.
 * @param grad      The gradient image, such that we can skip non-gradient
 * pixels
 * @param gpc       The output image of 32bit codes
 * @param fastmask  The fastmask containing the gpc filter
 * @param idx       The gradient indices. Only used if no intrincs are available
 *                  and the call gets forwarded to the naive implementation.
 * @param width     The width of the image at pointer *in
 * @param height    The height of the image at pointer *in
 */
void gpcFilterNaive(uint8_t* in,
                    const uint8_t* grad,
                    uint32_t* gpc,
                    std::vector<int32_t> fastmask,
                    std::vector<int>& idx,
                    int width,
                    int height);

/**
 * @brief Applies a gpc filter defined by the pixel-difference tests in
 * fastmask. Additionally uses a threshold vector (tau) Naive implementation.
 *
 * @param in        The input image.
 * @param grad      The gradient image, such that we can skip non-gradient
 * pixels
 * @param gpc       The output image of 32bit codes
 * @param fastmask  The fastmask containing the gpc filter
 * @param width     The width of the image at pointer *in
 * @param height    The height of the image at pointer *in
 */
void gpcFilterTauNaive(uint8_t* in,
                       const uint8_t* grad,
                       uint32_t* gpc,
                       std::vector<int32_t> fastmask,
                       std::vector<int> tau,
                       std::vector<int>& idx,
                       int width,
                       int height);
/**
   * @brief      boxfilter using SSE2 instructions. Loosely based on
   *             https://www.ignorantus.com/box_sse2/, published under
   *             the https://creativecommons.org/publicdomain/zero/1.0/ licence.
   *
   * @param      in       input image
   * @param      blurred  The blurred
   * @param[in]  width    The width
   * @param[in]  height   The height
   * @param[in]  numThreads number of threads to use
   */
void box(uint8_t* in, uint8_t* blurred, int width, int height, int numThreads);

/**
 * @brief      3x3 Sobel filter. Input dimension must be multiple of 16
 *
 * @param      in         { parameter_description }
 * @param      blurred    The blurred
 * @param[in]  width      The width
 * @param[in]  height     The height
 * @param[in]  threshold  The threshold
 * @param[in]  numThreads number of threads to use
 */

void sobel(uint8_t* in,
           uint8_t* blurred,
           int width,
           int height,
           uint8_t threshold,
           int numThreads);

/**
 * @brief Checks if the 128bits in xmm are all zero
 *
 * @param xmm
 *
 * @return true if all zeros, false otherwise
 */
#ifdef _INTRINSICS_SSE
bool isAllZeros(__m128i xmm);
#endif
/**
 * @brief Applies a gpc filter defined by the pixel-difference tests in
 * fastmask. Accelerated with SSE.
 *
 * @param in        The input image.
 * @param grad      The gradient image, such that we can skip non-gradient
 * pixels
 * @param gpc       The output image of 32bit codes
 * @param fastmask  The fastmask containing the gpc filter
 * @param idx       The gradient indices. Only used if no intrincs are available
 *                  and the call gets forwarded to the naive implementation.
 * @param width     The width of the image at pointer *in
 * @param height    The height of the image at pointer *in
 * @param numThreadsNumber of threads to use
 */
void gpcFilter(uint8_t* in,
               const uint8_t* grad,
               uint32_t* gpc,
               std::vector<int32_t> fastmask,
               std::vector<int>& idx,
               int width,
               int height,
               int numThreads);

/**
 * @brief Applies a gpc filter defined by the pixel-difference tests in
 * fastmask. Additionally uses a threshold vector (tau)
 *
 * @param in        The input image.
 * @param grad      The gradient image, such that we can skip non-gradient
 * pixels
 * @param gpc       The output image of 32bit codes
 * @param fastmask  The fastmask containing the gpc filter
 * @param width     The width of the image at pointer *in
 * @param height    The height of the image at pointer *in
 * @param numThreads Number of threads to use
 */
void gpcFilterTau(uint8_t* in,
                  const uint8_t* grad,
                  uint32_t* gpc,
                  std::vector<int32_t> fastmask,
                  std::vector<int> tau,
                  std::vector<int>& idx,
                  int width,
                  int height,
                  int numThreads); 
/**
 * @brief Naive version of 5x5 census transoform
 *
 * @param in      Input image
 * @param census  32bit census transform output
 * @param width   Width of the image at *in pointer
 * @param height  Heiht of the image at *in pointer
 */
void census5x5Naive(uint8_t* in, uint32_t* census, int width, int height);


/**
 * @brief 5x5 dense census transform of input image. binary codes are returned
 * as a 32bit image
 *
 * @param in
 * @param census
 * @param width
 * @param height
 */
void census5x5(uint8_t* in, uint32_t* census, int width, int height);
}  // namespace ndb
#endif
