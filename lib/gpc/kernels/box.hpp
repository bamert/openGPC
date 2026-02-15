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

#ifndef __NDB__KERNEL_BOX
#define __NDB__KERNEL_BOX
using namespace std;

#include "gpc/buffer.hpp"

namespace ndb {
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


}
#endif
