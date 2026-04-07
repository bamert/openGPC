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

#ifndef __NDB__KERNEL_SOBEL
#define __NDB__KERNEL_SOBEL
#include "gpc/buffer.hpp"

namespace ndb {
#if HWY_TARGET == HWY_AVX2
void sobelSSE(const uint8_t* in,
              uint8_t* blurred,
              int width,
              int start,
              int end,
              uint8_t threshold);

#endif
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
}  // namespace ndb
#endif
