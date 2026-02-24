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

#ifndef __NDB__KERNEL_GPC
#define __NDB__KERNEL_GPC
using namespace std;

#include "gpc/buffer.hpp"

namespace ndb {
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
 */
void gpcFilter(uint8_t* in,
               const uint8_t* grad,
               uint32_t* gpc,
               std::vector<int32_t> fastmask,
               std::vector<int>& idx,
               int width,
               int height);


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
 * fastmask. Additionally uses a threshold vector (tau)
 *
 * @param in        The input image.
 * @param grad      The gradient image, such that we can skip non-gradient
 * pixels
 * @param gpc       The output image of 32bit codes
 * @param fastmask  The fastmask containing the gpc filter
 * @param width     The width of the image at pointer *in
 * @param height    The height of the image at pointer *in
 */
void gpcFilterTau(uint8_t* in,
                  const uint8_t* grad,
                  uint32_t* gpc,
                  std::vector<int32_t> fastmask,
                  std::vector<int> tau,
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
 * @brief Checks if the 128bits in xmm are all zero
 *
 * @param xmm
 *
 * @return true if all zeros, false otherwise
 */
#if (HWY_ARCH_X86) && (HWY_TARGET == HWY_AVX2)
bool isAllZeros(__m128i xmm);
void gpcFilterTauSSE(uint8_t* in,
                  const uint8_t* grad,
                  uint32_t* gpc,
                  std::vector<int32_t> fastmask,
                  std::vector<int> tau,
                  std::vector<int>& idx,
                  int width,
                  int height);
void gpcFilterSSE(uint8_t* in,
               const uint8_t* grad,
               uint32_t* gpc,
               std::vector<int32_t> fastmask,
               std::vector<int>& idx,
               int width,
               int height);


#endif


}
#endif
