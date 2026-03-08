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
// Implements and extends the method proposed in
// The Global Patch Collider
// Shenlong Wang, Sean Ryan Fanello, Christoph Rhemann, Shahram Izadi, Pushmeet
// Kohli CVPR 2016 Code Author: Niklaus Bamert (bamertn@ethz.ch)
#include <Eigen/Dense>
//#include <arm_neon.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

// GPC includes
#include "gpc/Feature.hpp"
#include "gpc/SintelOpticalFlow.hpp"
#include "gpc/SintelStereo.hpp"
#include "gpc/buffer.hpp"
#include "gpc/kernels/sobel.hpp"
#include "gpc/kernels/box.hpp"
#include "gpc/kernels/gpc.hpp"
#include "gpc/kernels/utils.hpp"
#include "gpc/hashmatch.hpp"
#include "gpc/forest.hpp"


namespace gpc {
namespace inference {
void Forest::prepareSoAFramesPersistentSingleSlab(
    std::vector<ndb::Descriptor>& srcStates,
    std::vector<ndb::Descriptor>& tarStates,
    SoAFramePersistentSingleSlab& srcFrame, 
    SoAFramePersistentSingleSlab& tarFrame) {

    uint32_t srcCounts[256] = {0}, tarCounts[256] = {0};
    for (const auto& s : srcStates) srcCounts[s.state & 0xFF]++;
    for (const auto& t : tarStates) tarCounts[t.state & 0xFF]++;

    StateIdx* sP = srcFrame.slab.data();
    StateIdx* tP = tarFrame.slab.data();
    for (int i = 0; i < 256; ++i) {
        srcFrame.bucketData[i] = sP;
        srcFrame.bucketSizes[i] = srcCounts[i];
        tarFrame.bucketData[i] = tP;
        tarFrame.bucketSizes[i] = tarCounts[i];
        sP += srcCounts[i]; tP += tarCounts[i];
    }

    uint32_t sW[256] = {0}, tW[256] = {0};
    for (uint32_t i = 0; i < (uint32_t)srcStates.size(); ++i) {
        uint64_t sv = srcStates[i].state;
        uint64_t tv = tarStates[i].state;
        srcFrame.bucketData[sv & 0xFF][sW[sv & 0xFF]++] = {sv, i};
        tarFrame.bucketData[tv & 0xFF][tW[tv & 0xFF]++] = {tv, i};
    }
}
void Forest::prepareSoAFramesPersistent(
    std::vector<ndb::Descriptor>& srcStates,
    std::vector<ndb::Descriptor>& tarStates,
    SoAFramePersistent& srcFrame, 
    SoAFramePersistent& tarFrame) {
    assert(srcStates.size() == tarStates.size());
    assert(srcStates.size() <= 256 * 16384); // limit for max unique items in our table design
/*
 // This is only slightly slower than the bit below.
    const uint32_t BUCKET_COUNT = 256;
    const uint64_t BUCKET_MASK = 0xFF;

    // 1. Histogram (To find bucket boundaries)
    uint32_t srcCounts[BUCKET_COUNT] = {0};
    uint32_t tarCounts[BUCKET_COUNT] = {0};
    for (const auto& s : srcStates) srcCounts[s.state & BUCKET_MASK]++;
    for (const auto& t : tarStates) tarCounts[t.state & BUCKET_MASK]++;

    // 2. Setup Bucket Pointers into the Slab
    // We treat the slab like a custom allocator
    uint64_t* srcPtr = srcFrame.statesSlab.data();
    uint32_t* srcIdxPtr = srcFrame.indicesSlab.data();
    uint64_t* tarPtr = tarFrame.statesSlab.data();
    uint32_t* tarIdxPtr = tarFrame.indicesSlab.data();

    for (uint32_t i = 0; i < BUCKET_COUNT; ++i) {
        srcFrame.bucketStates[i] = srcPtr;
        srcFrame.bucketIndices[i] = srcIdxPtr;
        srcFrame.bucketSizes[i] = srcCounts[i];
        
        tarFrame.bucketStates[i] = tarPtr;
        tarFrame.bucketIndices[i] = tarIdxPtr;
        tarFrame.bucketSizes[i] = tarCounts[i];

        srcPtr += srcCounts[i];
        srcIdxPtr += srcCounts[i];
        tarPtr += tarCounts[i];
        tarIdxPtr += tarCounts[i];
    }

    // 3. The "Pure Scatter" (No push_back, no resize, no zeroing)
    uint32_t srcWriteIdx[BUCKET_COUNT] = {0};
    uint32_t tarWriteIdx[BUCKET_COUNT] = {0};

    for (uint32_t i = 0; i < (uint32_t)srcStates.size(); ++i) {
        uint64_t s = srcStates[i].state;
        uint32_t b = s & BUCKET_MASK;
        uint32_t pos = srcWriteIdx[b]++;
        srcFrame.bucketStates[b][pos] = s;
        srcFrame.bucketIndices[b][pos] = i;
    }

    for (uint32_t i = 0; i < (uint32_t)tarStates.size(); ++i) {
        uint64_t s = tarStates[i].state;
        uint32_t b = s & BUCKET_MASK;
        uint32_t pos = tarWriteIdx[b]++;
        tarFrame.bucketStates[b][pos] = s;
        tarFrame.bucketIndices[b][pos] = i;
    }
    */
    const uint32_t BUCKET_COUNT = 256;
    const uint64_t BUCKET_MASK = 0xFF;

    uint32_t srcCounts[BUCKET_COUNT] = {0};
    uint32_t tarCounts[BUCKET_COUNT] = {0};

    // 1. Fused Histogram Pass (Assuming equal sizes as per your note)
    const uint32_t totalSize = (uint32_t)srcStates.size();
    for (uint32_t i = 0; i < totalSize; ++i) {
        srcCounts[srcStates[i].state & BUCKET_MASK]++;
        tarCounts[tarStates[i].state & BUCKET_MASK]++;
    }

    // 2. Setup Bucket Pointers (Unchanged, this is fast)
    uint64_t* sP = srcFrame.statesSlab.data();
    uint32_t* sI = srcFrame.indicesSlab.data();
    uint64_t* tP = tarFrame.statesSlab.data();
    uint32_t* tI = tarFrame.indicesSlab.data();

    for (uint32_t i = 0; i < BUCKET_COUNT; ++i) {
        srcFrame.bucketStates[i] = sP;
        srcFrame.bucketIndices[i] = sI;
        srcFrame.bucketSizes[i] = srcCounts[i];
        tarFrame.bucketStates[i] = tP;
        tarFrame.bucketIndices[i] = tI;
        tarFrame.bucketSizes[i] = tarCounts[i];
        sP += srcCounts[i]; sI += srcCounts[i];
        tP += tarCounts[i]; tI += tarCounts[i];
    }

    // 3. Optimized Fused Scatter
    uint32_t srcWriteIdx[BUCKET_COUNT] = {0};
    uint32_t tarWriteIdx[BUCKET_COUNT] = {0};

    // Unroll by 2 to keep the M3's execution ports saturated
    uint32_t i = 0;
    for (; i + 1 < totalSize; i += 2) {
        // Source pair
        uint64_t s0 = srcStates[i].state;
        uint64_t s1 = srcStates[i+1].state;
        uint32_t bS0 = s0 & BUCKET_MASK;
        uint32_t bS1 = s1 & BUCKET_MASK;

        srcFrame.bucketStates[bS0][srcWriteIdx[bS0]++] = s0;
        srcFrame.bucketIndices[bS0][srcWriteIdx[bS0]-1] = i;
        srcFrame.bucketStates[bS1][srcWriteIdx[bS1]++] = s1;
        srcFrame.bucketIndices[bS1][srcWriteIdx[bS1]-1] = i+1;

        // Target pair
        uint64_t t0 = tarStates[i].state;
        uint64_t t1 = tarStates[i+1].state;
        uint32_t bT0 = t0 & BUCKET_MASK;
        uint32_t bT1 = t1 & BUCKET_MASK;

        tarFrame.bucketStates[bT0][tarWriteIdx[bT0]++] = t0;
        tarFrame.bucketIndices[bT0][tarWriteIdx[bT0]-1] = i;
        tarFrame.bucketStates[bT1][tarWriteIdx[bT1]++] = t1;
        tarFrame.bucketIndices[bT1][tarWriteIdx[bT1]-1] = i+1;
    }

    // Handle remainder
    for (; i < totalSize; ++i) {
        uint64_t s = srcStates[i].state;
        uint32_t bS = s & BUCKET_MASK;
        srcFrame.bucketStates[bS][srcWriteIdx[bS]++] = s;
        srcFrame.bucketIndices[bS][srcWriteIdx[bS]-1] = i;

        uint64_t t = tarStates[i].state;
        uint32_t bT = t & BUCKET_MASK;
        tarFrame.bucketStates[bT][tarWriteIdx[bT]++] = t;
        tarFrame.bucketIndices[bT][tarWriteIdx[bT]-1] = i;
    }
}

 // Here we did allocation within the prepare. we can move that part out
std::pair<SoAFrame, SoAFrame> Forest::prepareSoAFrames(
    std::vector<ndb::Descriptor>& srcStates,
    std::vector<ndb::Descriptor>& tarStates) {
    SoAFrame srcFrame, tarFrame;
    srcFrame.reserve(srcStates.size());
    tarFrame.reserve(tarStates.size());

    const uint64_t MASK = 0xFF;

    // Distribute into buckets based on the last 8 bits of the state
    for (uint32_t i = 0; i < srcStates.size(); ++i) {
        uint64_t s = srcStates[i].state;
        srcFrame.states[s & MASK].push_back(s);
        srcFrame.indices[s & MASK].push_back(i);
    }

    for (uint32_t i = 0; i < tarStates.size(); ++i) {
        uint64_t s = tarStates[i].state;
        tarFrame.states[s & MASK].push_back(s);
        tarFrame.indices[s & MASK].push_back(i);
    }

    return {srcFrame, tarFrame};
}
void Forest::matchPipelinedBranchlessPreallocateSingleSlab(
    SoAFramePersistentSingleSlab& src, SoAFramePersistentSingleSlab& tar,
    std::vector<uint32_t>& outS, std::vector<uint32_t>& outT) {

    struct Slot { 
        uint64_t key;   // The 64-bit Descriptor/State ID
        uint32_t idx;   // The original global index in the Source array
        uint32_t gen;   // The "Generation" ID (replaces memset/clear)
        uint32_t count; // The match state (0=empty, 1=unique, >1=dup, 0xFF..=matched)
    };
    static std::vector<Slot> table(16384, {0, 0, 0, 0});
    static uint32_t currentGen = 1;

    for (int b = 0; b < 256; ++b) {
        StateIdx* sData = src.bucketData[b];
        uint32_t  sSize = src.bucketSizes[b];
        if (sSize == 0) continue;

        const uint32_t mask = (sSize < 1000) ? 2047 : 16383;
        const uint32_t shift = (sSize < 1000) ? 53 : 50;
        currentGen++;

        for (uint32_t i = 0; i < sSize; ++i) {
            uint64_t k = sData[i].state;
            uint32_t h = (k * 11400714819323198485llu) >> shift;
            h &= mask;
            while (table[h].gen == currentGen && table[h].key != k) h = (h + 1) & mask;
            if (table[h].gen != currentGen) table[h] = {k, sData[i].index, currentGen, 1};
            else table[h].count++;
        }

        StateIdx* tData = tar.bucketData[b];
        uint32_t  tSize = tar.bucketSizes[b];
        for (uint32_t i = 0; i < tSize; ++i) {
            uint64_t k = tData[i].state;
            uint32_t h = (k * 11400714819323198485llu) >> shift;
            h &= mask;
            while (table[h].gen == currentGen && table[h].key != k) h = (h + 1) & mask;

            if (table[h].gen == currentGen && table[h].key == k) {
                if (table[h].count == 1) {
                    outS.push_back(table[h].idx);
                    outT.push_back(tData[i].index);
                    table[h].count = 0xFFFFFFFF;
                } else if (table[h].count == 0xFFFFFFFF) {
                    outS.pop_back(); outT.pop_back();
                    table[h].count = 0xEEEEEEEE;
                }
            }
        }
    }
}
/*
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> Forest::matchAdaptiveNeon(
    SoAFrame& src, 
    SoAFrame& tar) {

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> result;
    result.first.reserve(10000); 
    result.second.reserve(10000);

    // Slot is exactly 32 bytes. 2 Slots = 64 bytes (1 Cache Line).
    struct alignas(16) Slot { 
        uint64_t key; 
        uint32_t idx; 
        uint32_t gen;   
        uint32_t count; 
        uint32_t padding; 
    };

    static uint32_t currentGen = 1;
    static std::vector<Slot> table(8192, {0, 0, 0, 0, 0});

    for (int b = 0; b < 256; ++b) {
        const auto& sStates = src.states[b];
        const auto& sIdxs   = src.indices[b];
        if (sStates.empty()) continue;

        const uint32_t mask = (sStates.size() < 500) ? 1023 : 8191;
        currentGen++;

        // --- PART 1: SOURCE FILL (Keep Scalar as it's usually not the bottleneck) ---
        for (size_t i = 0; i < sStates.size(); ++i) {
            uint64_t k = sStates[i];
            uint32_t h = (k * 11400714819323198485llu) >> (64 - 13);
            h &= mask;

            while (table[h].gen == currentGen && table[h].key != k) {
                h = (h + 1) & mask;
            }
            
            if (table[h].gen != currentGen) {
                table[h] = {k, sIdxs[i], currentGen, 1, 0};
            } else {
                table[h].count++;
            }
        }

        const auto& tStates = tar.states[b];
        const auto& tIdxs   = tar.indices[b];

        // --- PART 2: TARGET MATCH (NEON Vectorized Window) ---
        uint64x2_t genVec = vdupq_n_u64((uint64_t)currentGen << 32); // Gen is at offset 12 in slot
        
        for (size_t i = 0; i < tStates.size(); ++i) {
            uint64_t k = tStates[i];
            uint32_t h = (k * 11400714819323198485llu) >> (64 - 13);
            h &= mask;

            uint64x2_t targetKeyV = vdupq_n_u64(k);
            bool found = false;

            // Check 2 slots at a time (One Cache Line)
            // This loop usually terminates in the first iteration (h and h+1)
            while (true) {
                // Load keys from Slot H and Slot H+1
                // We use vld2 to pick the 'key' field which is the first 8 bytes of each 32-byte slot
                // For simplicity and speed on M3, we'll just do direct pointer access:
                uint64_t k0 = table[h].key;
                uint64_t k1 = table[(h + 1) & mask].key;
                uint32_t g0 = table[h].gen;
                uint32_t g1 = table[(h + 1) & mask].gen;

                uint64x2_t keysV = {k0, k1};
                uint32x2_t gensV = {g0, g1};

                // Compare keys
                uint64x2_t keyMatch = vceqq_u64(keysV, targetKeyV);
                // Compare generations
                uint32x2_t genMatch = vceq_u32(gensV, vdup_n_u32(currentGen));

                // Check lane 0
                if (vgetq_lane_u64(keyMatch, 0) && vget_lane_u32(genMatch, 0)) {
                    if (table[h].count == 1) {
                        result.first.push_back(table[h].idx);
                        result.second.push_back(tIdxs[i]);
                        table[h].count = 0xFFFFFFFF;
                    } else if (table[h].count == 0xFFFFFFFF) {
                        result.first.pop_back(); result.second.pop_back();
                        table[h].count = 0xEEEEEEEE;
                    }
                    found = true; break;
                }
                
                // Check lane 1
                uint32_t nextH = (h + 1) & mask;
                if (vgetq_lane_u64(keyMatch, 1) && vget_lane_u32(genMatch, 1)) {
                    if (table[nextH].count == 1) {
                        result.first.push_back(table[nextH].idx);
                        result.second.push_back(tIdxs[i]);
                        table[nextH].count = 0xFFFFFFFF;
                    } else if (table[nextH].count == 0xFFFFFFFF) {
                        result.first.pop_back(); result.second.pop_back();
                        table[nextH].count = 0xEEEEEEEE;
                    }
                    found = true; break;
                }

                // If neither matches and both are "current", we must keep probing
                if (g0 == currentGen && g1 == currentGen) {
                    h = (h + 2) & mask;
                } else {
                    // One of them is an empty slot (gen != currentGen), stop searching
                    break;
                }
            }
        }
    }
    return result;
}
*/
void Forest::matchPipelinedBranchlessPreallocate(
    SoAFramePersistent& src, 
    SoAFramePersistent& tar,
    std::vector<uint32_t>& resultSrc,
    std::vector<uint32_t>& resultTar) {

    //std::pair<std::vector<uint32_t>, std::vector<uint32_t>> result;
    // For 100M items, we might find more matches; 
    // adjusting reserve to prevent mid-run reallocations.
    //result.first.reserve(src.statesSlab.size() / 100); 
    //result.second.reserve(src.statesSlab.size() / 100);

    struct Slot { 
        uint64_t key; 
        uint32_t idx; 
        uint32_t gen;   
        uint32_t count; 
    };

    static uint32_t currentGen = 1; 
    // Increased table size slightly to 16k to further reduce Pareto collisions
    static std::vector<Slot> table(16384, {0, 0, 0, 0}); 

    for (int b = 0; b < 256; ++b) {
        uint64_t* sStates = src.bucketStates[b];
        uint32_t* sIdxs   = src.bucketIndices[b];
        uint32_t  sSize   = src.bucketSizes[b];
        
        if (sSize == 0) continue;

        // Adaptive Mask: 2k for small, 16k for large
        const uint32_t mask = (sSize < 1000) ? 2047 : 16383;
        const uint32_t shift = (sSize < 1000) ? (64 - 11) : (64 - 14);
        currentGen++; 

        // 1. Fill Table (Source)
        for (size_t i = 0; i < sSize; ++i) {
            uint64_t k = sStates[i];
            uint32_t h = (k * 11400714819323198485llu) >> shift;
            h &= mask;

            // Branchless-ish Probe: Most IDs are unique, so this loop
            // is predicted "not taken" after the first check.
            while (table[h].gen == currentGen && table[h].key != k) {
                h = (h + 1) & mask;
            }
            
            if (table[h].gen != currentGen) {
                table[h] = {k, sIdxs[i], currentGen, 1};
            } else {
                table[h].count++; 
            }
        }

        // 2. Intersect (Target) with Software Pipelining
        uint64_t* tStates = tar.bucketStates[b];
        uint32_t* tIdxs   = tar.bucketIndices[b];
        uint32_t  tSize   = tar.bucketSizes[b];

        for (size_t i = 0; i < tSize; ++i) {
            // Manual prefetch of the state 16 elements ahead to stay in L1
            if (i + 16 < tSize) {
                __builtin_prefetch(&tStates[i + 16], 0, 3);
            }

            uint64_t k = tStates[i];
            uint32_t h = (k * 11400714819323198485llu) >> shift;
            h &= mask;

            // Probe logic
            while (table[h].gen == currentGen && table[h].key != k) {
                h = (h + 1) & mask;
            }

            if (table[h].gen == currentGen && table[h].key == k) {
                const uint32_t cnt = table[h].count;
                if (cnt == 1) {
                    resultSrc.push_back(table[h].idx);
                    resultTar.push_back(tIdxs[i]);
                    table[h].count = 0xFFFFFFFF; 
                } else if (cnt == 0xFFFFFFFF) {
                    // Pareto multi-match removal logic
                    resultSrc.pop_back();
                    resultTar.pop_back();
                    table[h].count = 0xEEEEEEEE; 
                }
            }
        }
    }
}
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> Forest::matchPipelinedBranchless(
    SoAFramePersistent& src, 
    SoAFramePersistent& tar) {

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> result;
    // For 100M items, we might find more matches; 
    // adjusting reserve to prevent mid-run reallocations.
    result.first.reserve(src.statesSlab.size() / 100); 
    result.second.reserve(src.statesSlab.size() / 100);

    struct Slot { 
        uint64_t key; 
        uint32_t idx; 
        uint32_t gen;   
        uint32_t count; 
    };

    static uint32_t currentGen = 1; 
    // Increased table size slightly to 16k to further reduce Pareto collisions
    static std::vector<Slot> table(16384, {0, 0, 0, 0}); 

    for (int b = 0; b < 256; ++b) {
        uint64_t* sStates = src.bucketStates[b];
        uint32_t* sIdxs   = src.bucketIndices[b];
        uint32_t  sSize   = src.bucketSizes[b];
        
        if (sSize == 0) continue;

        // Adaptive Mask: 2k for small, 16k for large
        const uint32_t mask = (sSize < 1000) ? 2047 : 16383;
        const uint32_t shift = (sSize < 1000) ? (64 - 11) : (64 - 14);
        currentGen++; 

        // 1. Fill Table (Source)
        for (size_t i = 0; i < sSize; ++i) {
            uint64_t k = sStates[i];
            uint32_t h = (k * 11400714819323198485llu) >> shift;
            h &= mask;

            // Branchless-ish Probe: Most IDs are unique, so this loop
            // is predicted "not taken" after the first check.
            while (table[h].gen == currentGen && table[h].key != k) {
                h = (h + 1) & mask;
            }
            
            if (table[h].gen != currentGen) {
                table[h] = {k, sIdxs[i], currentGen, 1};
            } else {
                table[h].count++; 
            }
        }

        // 2. Intersect (Target) with Software Pipelining
        uint64_t* tStates = tar.bucketStates[b];
        uint32_t* tIdxs   = tar.bucketIndices[b];
        uint32_t  tSize   = tar.bucketSizes[b];

        for (size_t i = 0; i < tSize; ++i) {
            // Manual prefetch of the state 16 elements ahead to stay in L1
            if (i + 16 < tSize) {
                __builtin_prefetch(&tStates[i + 16], 0, 3);
            }

            uint64_t k = tStates[i];
            uint32_t h = (k * 11400714819323198485llu) >> shift;
            h &= mask;

            // Probe logic
            while (table[h].gen == currentGen && table[h].key != k) {
                h = (h + 1) & mask;
            }

            if (table[h].gen == currentGen && table[h].key == k) {
                const uint32_t cnt = table[h].count;
                if (cnt == 1) {
                    result.first.push_back(table[h].idx);
                    result.second.push_back(tIdxs[i]);
                    table[h].count = 0xFFFFFFFF; 
                } else if (cnt == 0xFFFFFFFF) {
                    // Pareto multi-match removal logic
                    result.first.pop_back();
                    result.second.pop_back();
                    table[h].count = 0xEEEEEEEE; 
                }
            }
        }
    }
    return result;
}
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> Forest::matchAdaptivePersistent(
    SoAFramePersistent& src, 
    SoAFramePersistent& tar) {

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> result;
    result.first.reserve(10000); 
    result.second.reserve(10000);

    struct Slot { 
        uint64_t key; 
        uint32_t idx; 
        uint32_t gen;   // Generation counter
        uint32_t count; // 1=SrcUnique, 0xFFFFFFFF=Matched, etc.
    };

    static uint32_t currentGen = 1; 
    static std::vector<Slot> table(8192, {0, 0, 0, 0}); 

    for (int b = 0; b < 256; ++b) {
        uint64_t* sStates = src.bucketStates[b];
        uint32_t* sIdxs   = src.bucketIndices[b];
        uint32_t  sSize   = src.bucketSizes[b];
        
        if (sSize == 0) continue;

        const uint32_t mask = (sSize < 500) ? 1023 : 8191;
        currentGen++; 

        // 1. Fill Table
        for (size_t i = 0; i < sSize; ++i) {
            uint64_t k = sStates[i];
            uint32_t h = (k * 11400714819323198485llu) >> (64 - 13);
            h &= mask;

            while (table[h].gen == currentGen && table[h].key != k) {
                h = (h + 1) & mask;
            }
            
            if (table[h].gen != currentGen) {
                table[h] = {k, sIdxs[i], currentGen, 1};
            } else {
                table[h].count++; 
            }
        }

        // 2. Intersect
        uint64_t* tStates = tar.bucketStates[b];
        uint32_t* tIdxs   = tar.bucketIndices[b];
        uint32_t  tSize   = tar.bucketSizes[b];

        for (size_t i = 0; i < tSize; ++i) {
            uint64_t k = tStates[i];
            uint32_t h = (k * 11400714819323198485llu) >> (64 - 13);
            h &= mask;

            while (table[h].gen == currentGen && table[h].key != k) {
                h = (h + 1) & mask;
            }

            if (table[h].gen == currentGen && table[h].key == k) {
                if (table[h].count == 1) {
                    result.first.push_back(table[h].idx);
                    result.second.push_back(tIdxs[i]);
                    table[h].count = 0xFFFFFFFF; 
                } else if (table[h].count == 0xFFFFFFFF) {
                    result.first.pop_back();
                    result.second.pop_back();
                    table[h].count = 0xEEEEEEEE; 
                }
            }
        }
    }
    return result;
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> Forest::matchAdaptive(
    SoAFrame& src, 
    SoAFrame& tar) {

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> result;
    result.first.reserve(10000); 
    result.second.reserve(10000);

    struct Slot { 
        uint64_t key; 
        uint32_t idx; 
        uint32_t gen;   // Generation counter
        uint32_t count; // 1=SrcUnique, 0xFFFFFFFF=Matched, etc.
    };

    // Global generation for this call
    uint32_t currentGen = 1; 
    std::vector<Slot> table(8192, {0, 0, 0, 0}); 

    for (int b = 0; b < 256; ++b) {
        const auto& sStates = src.states[b];
        const auto& sIdxs   = src.indices[b];
        if (sStates.empty()) continue;

        // Adaptive Table Mask: Use smaller range for tiny buckets
        const uint32_t mask = (sStates.size() < 500) ? 1023 : 8191;
        currentGen++; 

        // 1. Fill Table
        for (size_t i = 0; i < sStates.size(); ++i) {
            // Prefetch an element roughly 16 iterations ahead (adjust based on testing)
            /*
             * This didn't help anymore. 
             * if (i + 16 < sStates.size()) {
                __builtin_prefetch(&sStates[i + 16], 0, 3);
                __builtin_prefetch(&sIdxs[i + 16], 0, 3);
            }*/
            uint64_t k = sStates[i];
            uint32_t h = (k * 11400714819323198485llu) >> (64 - 13);
            h &= mask;

            // Probe: Valid if gen matches AND key is different
            while (table[h].gen == currentGen && table[h].key != k) {
                h = (h + 1) & mask;
            }
            
            if (table[h].gen != currentGen) {
                table[h] = {k, sIdxs[i], currentGen, 1};
            } else {
                table[h].count++; // Duplicate in Source
            }
        }

        // 2. Intersect
        const auto& tStates = tar.states[b];
        const auto& tIdxs   = tar.indices[b];
        for (size_t i = 0; i < tStates.size(); ++i) {
            uint64_t k = tStates[i];
            uint32_t h = (k * 11400714819323198485llu) >> (64 - 13);
            h &= mask;

            while (table[h].gen == currentGen && table[h].key != k) {
                h = (h + 1) & mask;
            }

            if (table[h].gen == currentGen && table[h].key == k) {
                if (table[h].count == 1) {
                    result.first.push_back(table[h].idx);
                    result.second.push_back(tIdxs[i]);
                    table[h].count = 0xFFFFFFFF; 
                } else if (table[h].count == 0xFFFFFFFF) {
                    result.first.pop_back();
                    result.second.pop_back();
                    table[h].count = 0xEEEEEEEE; 
                }
            }
        }
    }
    return result;
}
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> Forest::matchBlockedBloom(
    SoAFrame& src, 
    SoAFrame& tar) {

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> result;
    result.first.reserve(10000); 
    result.second.reserve(10000);

    struct Slot { 
        uint64_t key; 
        uint32_t idx; 
        uint32_t count; 
    };

    const uint32_t TABLE_SIZE = 8192;
    const uint32_t HASH_MASK = TABLE_SIZE - 1;
    std::vector<Slot> table(TABLE_SIZE); 

    // A 512-bit Bloom Filter fits in exactly one Cache Line (64 bytes).
    // We use 8 x 64-bit integers to represent the 512 bits.
    uint64_t bloom[8];

    for (int b = 0; b < 256; ++b) {
        std::fill(table.begin(), table.end(), Slot{0, 0, 0});
        std::memset(bloom, 0, sizeof(bloom));

        const auto& sStates = src.states[b];
        const auto& sIdxs   = src.indices[b];
        const auto& tStates = tar.states[b];
        const auto& tIdxs   = tar.indices[b];

        // 1. Fill Table + Bloom Filter
        for (size_t i = 0; i < sStates.size(); ++i) {
            uint64_t k = sStates[i];
            
            // Set Bloom bit: use a different hash or shift for the bloom index
            // We'll use bits from the key to pick one of 512 bits
            uint32_t bHash = (k ^ (k >> 32));
            bloom[(bHash >> 6) & 7] |= (1ull << (bHash & 63));

            uint32_t h = (k * 11400714819323198485llu) >> (64 - 13); 
            h &= HASH_MASK;

            while (table[h].count > 0 && table[h].key != k) {
                h = (h + 1) & HASH_MASK;
            }
            
            table[h].key = k;
            table[h].idx = sIdxs[i];
            table[h].count++; 
        }

        // 2. Intersection with Bloom Filter Gate
        for (size_t i = 0; i < tStates.size(); ++i) {
            uint64_t k = tStates[i];
            
            // --- BLOOM FILTER GATE ---
            uint32_t bHash = (k ^ (k >> 32));
            if (!(bloom[(bHash >> 6) & 7] & (1ull << (bHash & 63)))) {
                continue; // 100% certainly not in Source. Skip hash probe!
            }
            // -------------------------

            uint32_t h = (k * 11400714819323198485llu) >> (64 - 13);
            h &= HASH_MASK;

            while (table[h].count > 0 && table[h].key != k) {
                h = (h + 1) & HASH_MASK;
            }

            if (table[h].key == k) {
                if (table[h].count == 1) {
                    result.first.push_back(table[h].idx);
                    result.second.push_back(tIdxs[i]);
                    table[h].count = 0xFFFFFFFF; 
                } else if (table[h].count == 0xFFFFFFFF) {
                    result.first.pop_back();
                    result.second.pop_back();
                    table[h].count = 0xEEEEEEEE; 
                }
            }
        }
    }
    return result;
}
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> Forest::matchParallelRadixPartitioning(
    SoAFrame& src, 
    SoAFrame& tar) {

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> result;
    result.first.reserve(10000); 
    result.second.reserve(10000);

    const uint32_t TABLE_SIZE = 8192;
    const uint32_t HASH_MASK = TABLE_SIZE - 1;
    
    // Aligned scratchpad to maximize L1/L2 cache efficiency
    struct alignas(64) Slot { 
        uint64_t key; 
        uint32_t idx; 
        uint32_t count; 
    };
    std::vector<Slot> table(TABLE_SIZE);

    for (int b = 0; b < 256; ++b) {
        // 1. FAST CLEAR
        // std::fill is optimized, but we only zero the 'count' to save cycles
        for(auto& s : table) s.count = 0;

        const auto& sStates = src.states[b];
        const auto& sIdxs   = src.indices[b];
        const size_t sSize  = sStates.size();

        // 2. PIPELINED FILL (Unrolled x4 for ILP)
        // We process 4 items at once to hide memory latency
        size_t i = 0;
        for (; i + 3 < sSize; i += 4) {
            for (int k = 0; k < 4; ++k) {
                uint64_t key = sStates[i + k];
                uint32_t h = (key * 11400714819323198485llu) >> (64 - 13);
                h &= HASH_MASK;

                while (table[h].count > 0 && table[h].key != key) h = (h + 1) & HASH_MASK;
                
                table[h].key = key;
                table[h].idx = sIdxs[i + k];
                table[h].count++;
            }
        }
        // Handle remainder
        for (; i < sSize; ++i) {
            uint64_t key = sStates[i];
            uint32_t h = (key * 11400714819323198485llu) >> (64 - 13);
            h &= HASH_MASK;
            while (table[h].count > 0 && table[h].key != key) h = (h + 1) & HASH_MASK;
            table[h].key = key; table[h].idx = sIdxs[i]; table[h].count++;
        }

        // 3. OPTIMISTIC INTERSECTION
        const auto& tStates = tar.states[b];
        const auto& tIdxs   = tar.indices[b];
        const size_t tSize  = tStates.size();

        for (size_t j = 0; j < tSize; ++j) {
            uint64_t key = tStates[j];
            uint32_t h = (key * 11400714819323198485llu) >> (64 - 13);
            h &= HASH_MASK;

            while (table[h].count > 0 && table[h].key != key) h = (h + 1) & HASH_MASK;

            if (table[h].key == key) {
                if (table[h].count == 1) {
                    result.first.push_back(table[h].idx);
                    result.second.push_back(tIdxs[j]);
                    table[h].count = 0xFFFFFFFF; // Mark as Matched
                } else if (table[h].count == 0xFFFFFFFF) {
                    // Pareto duplicate found in Target: Roll back
                    result.first.pop_back();
                    result.second.pop_back();
                    table[h].count = 0xEEEEEEEE; // Mark as Permanent Duplicate
                }
            }
        }
    }

    return result;
}
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> Forest::matchPreparedFramesFaster(
    SoAFrame& src, 
    SoAFrame& tar) {

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> result;
    result.first.reserve(10000); 
    result.second.reserve(10000);

    // Flat, cache-aligned slot structure
    struct Slot { 
        uint64_t key; 
        uint32_t idx; 
        uint32_t count; 
    };

    // 8192 slots = 128KB. This fits perfectly in your 4MB L2.
    // We use a power-of-two size to use bitwise AND instead of modulo %.
    const uint32_t TABLE_SIZE = 8192;
    const uint32_t HASH_MASK = TABLE_SIZE - 1;
    std::vector<Slot> table(TABLE_SIZE); 

    for (int b = 0; b < 256; ++b) {
        // FAST: std::fill is usually a vectorized memset.
        std::fill(table.begin(), table.end(), Slot{0, 0, 0});

        const auto& sStates = src.states[b];
        const auto& sIdxs   = src.indices[b];
        const auto& tStates = tar.states[b];
        const auto& tIdxs   = tar.indices[b];

        // 1. Fill Table from Source
        for (size_t i = 0; i < sStates.size(); ++i) {
            uint64_t k = sStates[i];
            // Fibonacci Hashing (very fast for 64-bit keys)
            uint32_t h = (k * 11400714819323198485llu) >> (64 - 13); 
            h &= HASH_MASK;

            while (table[h].count > 0 && table[h].key != k) {
                h = (h + 1) & HASH_MASK;
            }
            
            table[h].key = k;
            table[h].idx = sIdxs[i];
            table[h].count++; 
        }

        // 2. Secondary Uniqueness Check + Intersection
        // We reuse the 'count' field: 
        // 1 = Unique in Src
        // >1 = Duplicate in Src
        // 0 = Already Matched (prevents Target duplicates)
        for (size_t i = 0; i < tStates.size(); ++i) {
            uint64_t k = tStates[i];
            uint32_t h = (k * 11400714819323198485llu) >> (64 - 13);
            h &= HASH_MASK;

            while (table[h].count > 0 && table[h].key != k) {
                h = (h + 1) & HASH_MASK;
            }

            // We need to know if 'k' is unique in Target too.
            // A quick way is to check if it appears again in the target bucket.
            // For Pareto, we can use a "tombstone" logic:
            if (table[h].key == k) {
                if (table[h].count == 1) {
                    // This is the first time we see it in Target
                    result.first.push_back(table[h].idx);
                    result.second.push_back(tIdxs[i]);
                    table[h].count = 0xFFFFFFFF; // Mark as "Matched once"
                } else if (table[h].count == 0xFFFFFFFF) {
                    // Oh no, this is a Target duplicate! 
                    // We must remove the last added match.
                    result.first.pop_back();
                    result.second.pop_back();
                    table[h].count = 0xEEEEEEEE; // Mark as "Permanent Duplicate"
                }
            }
        }
    }
    return result;
}
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> Forest::matchPreparedFrames( SoAFrame& src, SoAFrame& tar) {

    // Initialize the pair of vectors
    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> result;
    
    // Heuristic: start with a reasonable reserve (e.g., 5% of average bucket size * 256)
    size_t initialReserve = (src.states[0].size() + tar.states[0].size()) * 6; 
    result.first.reserve(initialReserve);
    result.second.reserve(initialReserve);

    // Local structures for bucket-level uniqueness
    struct SrcInfo { uint32_t idx; bool isDup; };
    std::unordered_map<uint64_t, SrcInfo> bucketSrc;
    std::unordered_map<uint64_t, bool> bucketTar;

    for (int b = 0; b < 256; ++b) {
        bucketSrc.clear();
        bucketTar.clear();

        const auto& sStates = src.states[b];
        const auto& sIdxs   = src.indices[b];
        const auto& tStates = tar.states[b];
        const auto& tIdxs   = tar.indices[b];

        // 1. Process Source: Mark unique vs duplicates
        for (size_t i = 0; i < sStates.size(); ++i) {
            auto [it, inserted] = bucketSrc.try_emplace(sStates[i], SrcInfo{sIdxs[i], false});
            if (!inserted) it->second.isDup = true;
        }

        // 2. Process Target: Mark unique vs duplicates
        for (size_t i = 0; i < tStates.size(); ++i) {
            auto [it, inserted] = bucketTar.try_emplace(tStates[i], false);
            if (!inserted) it->second = true; // Mark as duplicate
        }

        // 3. Intersect unique-only IDs
        for (size_t i = 0; i < tStates.size(); ++i) {
            uint64_t id = tStates[i];
            
            // Check if unique in Target
            if (bucketTar[id] == false) {
                auto it = bucketSrc.find(id);
                // Check if exists in Source AND is unique there
                if (it != bucketSrc.end() && it->second.isDup == false) {
                    result.first.push_back(it->second.idx);
                    result.second.push_back(tIdxs[i]);
                }
            }
        }
    }

    return result;
}

    /**
     * @brief Computes sparse matches on a pair of rectified and smoothed
     * images. Here the src and tar images refer to the left and right images,
     * respectively.
     *
     * @param src    Preprocessed source(left) image
     * @param tar    Preprocessed target(right) image
     * @param fastmask    forest mask of relative integer offsets.
     *
     * @return
     */
std::vector<ndb::Correspondence> Forest::depthPriorFast(
    PreprocessedImage& src,
    PreprocessedImage& tar,
    FilterMask& fastmask,
    InferenceSettings& settings) {
    std::chrono::high_resolution_clock::time_point t0, t1;
    std::vector<ndb::Descriptor> statesSrc = evalFastMaskOnSubsetSSE(
        src.smooth, src.grad, src.mask, fastmask, settings);
    std::vector<ndb::Descriptor> statesTar = evalFastMaskOnSubsetSSE(
        tar.smooth, tar.grad, tar.mask, fastmask, settings);
    // Epipolar mode. Use upper 32bit of 64bit descriptor to store y
    // coordinate
    if (settings.epipolarMode_) {
        for (auto& el : statesSrc) el.state |= uint64_t(el.point.y) << 32;
        for (auto& el : statesTar) el.state |= uint64_t(el.point.y) << 32;
    }
    // Use sort method for matching
    if (settings.useHashtable_ == false) {
    t0 = sysTick();
        std::vector<ndb::Correspondence> corr =
            findCorrespondences(statesSrc, statesTar);
    t1 = sysTick();
    std::cout << "findCorrespondences (without allocation): " << gpc::inference::tickToMs(t1, t0) << " ms" << std::endl;
    std::cout << "length src: " << statesSrc.size() << std::endl;
        return corr;
    }
    // Use hashtable matching
    else {
        for (auto& q : statesSrc) q.srcDescr = true;
        for (auto& q : statesTar) q.srcDescr = false;

        ndb::Hashmatch<ndb::Descriptor> hm(
            214673,  // statesSrc.size() + statesTar.size() ,
            statesSrc.size() + statesTar.size());
        std::vector<std::pair<ndb::Descriptor, ndb::Descriptor>> corr;
        for (auto& q : statesSrc) hm.insert(q);
        for (auto& q : statesTar) hm.insert(q);
        hm.getDuplicates(corr);
        // Store vertices in a format that is more convenient for us:
        std::vector<ndb::Correspondence> corr2;
        for (auto& e : corr) {
            corr2.push_back(
                ndb::Correspondence(e.first.point, e.second.point));
        }

        return corr2;
    }
}
std::vector<ndb::Correspondence> Forest::findCorrespondences(
    std::vector<ndb::Descriptor>& srcStates,
    std::vector<ndb::Descriptor>& tarStates) {
    int numStates = std::min(srcStates.size(), tarStates.size());
    // Limit search to rectified epipolar case.
    std::sort(srcStates.begin(), srcStates.end());
    std::sort(tarStates.begin(), tarStates.end());
    std::vector<ndb::Correspondence> corr;
    uint32_t j = 0;
    for (uint32_t i = 0; i < srcStates.size(); ++i) {
        bool unique = true;
        while (i + 1 < srcStates.size() && srcStates[i] == srcStates[i + 1])
            ++i, unique = false;

        if (unique) {
            // emulates std::lowerbound behavior for arrays
            for (; j < tarStates.size() - 1; ++j) {
                if (!(tarStates[j] < srcStates[i])) break;
            }

            if (j != tarStates.size() - 1 && tarStates[j] == srcStates[i] &&
                ((j + 1) == tarStates.size() - 1 ||
                 !(tarStates[j] == tarStates[j + 1])))
                corr.push_back(ndb::Correspondence(srcStates[i].point,
                                                   tarStates[j].point));
        }
    }
    return corr;
}
#include <unordered_map>

// State machine for our IDs
enum class State : uint8_t { Unseen = 0, SeenOnce = 1, Duplicate = 2 };

#include <vector>
#include <unordered_map>
#include <cstdint>

std::vector<ndb::Correspondence> Forest::findCorrespondencesHash(
    std::vector<ndb::Descriptor>& srcStates,
    std::vector<ndb::Descriptor>& tarStates) {

    // Tracking states: 0 = Unseen, 1 = SeenOnce, 2 = Duplicate
    enum class Occurence : uint8_t { Unseen = 0, SeenOnce = 1, Duplicate = 2 };

    // 1. Map Source IDs: State -> {OccurenceLevel, OriginalIndex}
    // Pre-allocating prevents expensive rehashes during the loop
    std::unordered_map<uint64_t, std::pair<Occurence, uint32_t>> srcMap;
    srcMap.reserve(srcStates.size());

    for (uint32_t i = 0; i < srcStates.size(); ++i) {
        auto& entry = srcMap[srcStates[i].state];
        if (entry.first == Occurence::Unseen) {
            entry = {Occurence::SeenOnce, i};
        } else {
            entry.first = Occurence::Duplicate;
        }
    }

    // 2. Map Target IDs: State -> OccurenceLevel
    std::unordered_map<uint64_t, Occurence> tarMap;
    tarMap.reserve(tarStates.size());

    for (uint32_t j = 0; j < tarStates.size(); ++j) {
        auto& occ = tarMap[tarStates[j].state];
        if (occ == Occurence::Unseen) {
            occ = Occurence::SeenOnce;
        } else {
            occ = Occurence::Duplicate;
        }
    }

    // 3. Intersect unique pairs
    std::vector<ndb::Correspondence> corr;
    // Heuristic: Reserve 20% of the smaller set size for the results
    corr.reserve(std::min(srcStates.size(), tarStates.size()) / 5);

    for (uint32_t j = 0; j < tarStates.size(); ++j) {
        uint64_t currentID = tarStates[j].state;

        // Condition: Must be unique in Target AND unique in Source
        if (tarMap[currentID] == Occurence::SeenOnce) {
            auto it = srcMap.find(currentID);
            if (it != srcMap.end() && it->second.first == Occurence::SeenOnce) {
                // Correspondence(Point from Source, Point from Target)
                corr.push_back(ndb::Correspondence(
                    srcStates[it->second.second].point, 
                    tarStates[j].point
                ));
            }
        }
    }

    return corr;
}
#include <vector>
#include <cstdint>
#include <algorithm>

// A lightweight structure to avoid moving heavy Descriptor objects
struct KeyIndex {
    uint64_t state;
    uint32_t index;
};
#include <vector>
#include <cstdint>
#include <array>

std::vector<ndb::Correspondence> Forest::findCorrespondencesTurbo(
    std::vector<ndb::Descriptor>& srcStates,
    std::vector<ndb::Descriptor>& tarStates) {

    const int BUCKETS = 256;
    const uint64_t MASK = 0xFF;

    // --- STEP 1: Linear Partitioning (Radix Pass) ---
    // We use a single flat buffer to avoid 256 separate vector allocations
    std::vector<KeyIndex> srcBuffer(srcStates.size());
    std::vector<KeyIndex> tarBuffer(tarStates.size());
    std::array<size_t, BUCKETS> srcCounts = {0}, tarCounts = {0};
    std::array<size_t, BUCKETS> srcOffsets, tarOffsets;

    for (const auto& s : srcStates) srcCounts[s.state & MASK]++;
    for (const auto& t : tarStates) tarCounts[t.state & MASK]++;

    srcOffsets[0] = tarOffsets[0] = 0;
    for (int i = 1; i < BUCKETS; ++i) {
        srcOffsets[i] = srcOffsets[i - 1] + srcCounts[i - 1];
        tarOffsets[i] = tarOffsets[i - 1] + tarCounts[i - 1];
    }

    auto srcCursors = srcOffsets;
    auto tarCursors = tarOffsets;

    for (uint32_t i = 0; i < srcStates.size(); ++i) {
        srcBuffer[srcCursors[srcStates[i].state & MASK]++] = {srcStates[i].state, i};
    }
    for (uint32_t i = 0; i < tarStates.size(); ++i) {
        tarBuffer[tarCursors[tarStates[i].state & MASK]++] = {tarStates[i].state, i};
    }

    // --- STEP 2: In-Cache Hashing ---
    std::vector<ndb::Correspondence> corr;
    corr.reserve(std::min(srcStates.size(), tarStates.size()) / 8);

    // Using a tiny fixed-size hash table for each bucket to stay in L1/L2 cache
    // State: 0 = Unseen, 1 = SeenOnce, 2 = Duplicate
    struct LocalVal { uint32_t index; uint8_t count; };
    
    // We reuse this map across buckets to avoid reallocating
    // A simple open-addressed hash map for the bucket
    std::unordered_map<uint64_t, LocalVal> bucketMap;
    bucketMap.reserve(srcStates.size() / BUCKETS * 2);

    for (int b = 0; b < BUCKETS; ++b) {
        bucketMap.clear();

        // Load Source bucket into local cache-friendly map
        size_t srcStart = srcOffsets[b];
        size_t srcEnd = srcStart + srcCounts[b];
        for (size_t i = srcStart; i < srcEnd; ++i) {
            auto& entry = bucketMap[srcBuffer[i].state];
            entry.index = srcBuffer[i].index;
            entry.count = (entry.count == 0) ? 1 : 2;
        }

        // Intersect with Target bucket
        size_t tarStart = tarOffsets[b];
        size_t tarEnd = tarStart + tarCounts[b];
        
        // Secondary map to ensure target-side uniqueness
        std::unordered_map<uint64_t, uint8_t> tarUniqueness;
        for (size_t i = tarStart; i < tarEnd; ++i) {
            auto& count = tarUniqueness[tarBuffer[i].state];
            count = (count == 0) ? 1 : 2;
        }

        for (size_t i = tarStart; i < tarEnd; ++i) {
            uint64_t id = tarBuffer[i].state;
            if (tarUniqueness[id] == 1) {
                auto it = bucketMap.find(id);
                if (it != bucketMap.end() && it->second.count == 1) {
                    corr.push_back(ndb::Correspondence(
                        srcStates[it->second.index].point, 
                        tarStates[tarBuffer[i].index].point
                    ));
                }
            }
        }
    }

    return corr;
}
std::vector<ndb::Correspondence> Forest::findCorrespondencesHashingRadix(
    std::vector<ndb::Descriptor>& srcStates,
    std::vector<ndb::Descriptor>& tarStates) {

    const int NUM_BUCKETS = 256;
    const uint64_t MASK = 0xFF;

    // 1. Partition Source into Buckets
    std::vector<KeyIndex> srcBuckets[NUM_BUCKETS];
    for (int i = 0; i < NUM_BUCKETS; ++i) srcBuckets[i].reserve(srcStates.size() / NUM_BUCKETS * 1.2);
    
    for (uint32_t i = 0; i < srcStates.size(); ++i) {
        srcBuckets[srcStates[i].state & MASK].push_back({srcStates[i].state, i});
    }

    // 2. Partition Target into Buckets
    std::vector<uint64_t> tarBuckets[NUM_BUCKETS];
    for (int i = 0; i < NUM_BUCKETS; ++i) tarBuckets[i].reserve(tarStates.size() / NUM_BUCKETS * 1.2);
    
    for (uint32_t i = 0; i < tarStates.size(); ++i) {
        tarBuckets[tarStates[i].state & MASK].push_back(tarStates[i].state);
    }

    std::vector<ndb::Correspondence> corr;
    corr.reserve(std::min(srcStates.size(), tarStates.size()) / 5);

    // 3. Process each bucket pair
    // This part can be easily parallelized with #pragma omp parallel for
    for (int b = 0; b < NUM_BUCKETS; ++b) {
        if (srcBuckets[b].empty() || tarBuckets[b].empty()) continue;

        // Small local maps fit in L1/L2 Cache
        // Using a simple frequency map for the local bucket
        enum class Occ : uint8_t { Unseen = 0, SeenOnce = 1, Duplicate = 2 };
        
        struct LocalEntry {
            Occ occ = Occ::Unseen;
            uint32_t idx = 0;
        };

        // We use a flat hash map here. For simplicity in standard C++, 
        // std::unordered_map is used, but even it is faster here 
        // because it stays in cache.
        std::unordered_map<uint64_t, LocalEntry> localSrc;
        localSrc.reserve(srcBuckets[b].size());

        for (auto& ki : srcBuckets[b]) {
            auto& entry = localSrc[ki.state];
            if (entry.occ == Occ::Unseen) {
                entry = {Occ::SeenOnce, ki.index};
            } else {
                entry.occ = Occ::Duplicate;
            }
        }

        std::unordered_map<uint64_t, Occ> localTar;
        localTar.reserve(tarBuckets[b].size());
        for (uint64_t state : tarBuckets[b]) {
            auto& occ = localTar[state];
            occ = (occ == Occ::Unseen) ? Occ::SeenOnce : Occ::Duplicate;
        }

        // Intersect within the bucket
        // Since we are inside a bucket, we iterate the target indices
        // but we need to find the target point. 
        // To be fast, we'll re-scan the original tarStates for this bucket's IDs
        for (uint32_t j = 0; j < tarStates.size(); ++j) {
            uint64_t s = tarStates[j].state;
            if ((s & MASK) == b) { // Only process IDs belonging to this bucket
                if (localTar[s] == Occ::SeenOnce) {
                    auto it = localSrc.find(s);
                    if (it != localSrc.end() && it->second.occ == Occ::SeenOnce) {
                        corr.push_back(ndb::Correspondence(
                            srcStates[it->second.idx].point,
                            tarStates[j].point
                        ));
                    }
                }
            }
        }
    }

    return corr;
}
/**
 * @brief Evaluates a given forest mask on an image and returns the
 * descriptors
 *
 * @param img       The image
 * @param grad      gradient image
 * @param idx       offsets with high gradient pixels within the grad image
 * @param fastmask  the forest mask
 *
 * @return
 */
std::vector<ndb::Descriptor> Forest::evalFastMaskOnSubsetSSE(
    ndb::Buffer<uint8_t>& img,
    ndb::Buffer<uint8_t>& grad,
    std::vector<int>& idx,
    FilterMask& fastmask,
    InferenceSettings& settings) {
    std::chrono::high_resolution_clock::time_point t0, t1;

    // output buffer of same size
    ndb::Buffer<uint32_t> gpcstates(img.rows(), img.cols(), 0);
    if (fastmask.type == 0) {
        ndb::gpcFilter(img.data(),
                       grad.data(),
                       gpcstates.data(),
                       fastmask.mask,
                       idx,
                       img.cols(),
                       img.rows());
    } else {
        ndb::gpcFilterTau(img.data(),
                          grad.data(),
                          gpcstates.data(),
                          fastmask.mask,
                          fastmask.tau,
                          idx,
                          img.cols(),
                          img.rows());
    }
    std::vector<ndb::Descriptor> out(idx.size());
    int j = 0;

    for (auto k : idx) {
        int x = k % img.cols();
        int y = k / img.cols();
        out[j] = ndb::Descriptor(ndb::Point(x, y), gpcstates.data()[k]);
        j++;
    }
    return out;
}

/**
 * @brief Preprocesses an image. (smooth, binary sobel image and gradient
 * pixel indices)
 *
 * @param img     The raw input image to be preprocessed
 * @param InferenceSettings inference settings struct
 *
 * @return the preprocessed image
 */
PreprocessedImage Forest::preprocessImage(ndb::Buffer<uint8_t>& img,
                                  InferenceSettings settings) {
    assert((settings.gradientThreshold_ >= 0 &&
            settings.gradientThreshold_ <= 255) &&
           "gradientThreshold needs to be within 0...255");

    ndb::Buffer<uint8_t> smooth(img.rows(), img.cols());

    smooth.width = img.width;
    // 0.2ms
    ndb::box(img.data(),
             smooth.data(),
             img.cols(),
             img.rows(),
             settings.numThreads_);
    //4.2 *10^-5 ms
    smooth.clearBoundary();
    ndb::Buffer<uint8_t> grad(img.rows(), img.cols());
    grad.width = img.width;
    //4.2*10-5ms (unclear how)
    ndb::sobel(img.data(),
               grad.data(),
               img.cols(),
               img.rows(),
               settings.gradientThreshold_,
               settings.numThreads_);
    gpc::inference::time_point t0 = gpc::inference::sysTick();
    ndb::Buffer<int> idx;
    idx.resize(grad.rows(), grad.cols());
    auto ff = [&](ndb::Buffer<int>& in, std::vector<int>& out, int m) {
        for (int i = 0; i < m; i++) {
            int x = in.data()[i] % grad.cols();
            int y = in.data()[i] / grad.cols();
            if (y >= 13 && y < grad.rows() - 13 && x >= 13 &&
                x < grad.cols() - 13)
                out.push_back(in.data()[i]);
        }
    };
    int m;
    // mask indexing gradient pixels
    std::vector<int> mask;
    ndb::arr2ind(grad.data(), grad.cols() * grad.rows(), idx.data(), &m);
    ff(idx, mask, m);

    gpc::inference::time_point t1 = gpc::inference::sysTick();
    // Our outputs are: smooth, grad, mask;
    return PreprocessedImage(smooth, grad, mask);
}
/**
 * @brief Finds matches between two stereo images based on a given forest
 * mask.
 *
 * @param simg              source image (assumed to be the left image)
 * @param timg              target image (assumed to be the right image)
 * @param forestmask        forest mask, provided by readForest method
 * @param InferenceSettings inference settings struct
 * @return                  Set of correspondences (ptSrc, ptTar) where
 * ptSrc and ptTar are points in the source and target images, respectively.
 */
std::vector<ndb::Correspondence> Forest::stereoMatch(PreprocessedImage& simg,
                                             PreprocessedImage& timg,
                                             FilterMask& forestmask,
                                             InferenceSettings settings) {
    // make sure the delivered mask matches the image dimensions
    assert(
        (forestmask.width == simg.smooth.cols() &&
         forestmask.height == simg.smooth.rows()) &&
        "Source Image: dimension does not fit dimension of supplied forest "
        "mask");
    assert(
        (forestmask.width == timg.smooth.cols() &&
         forestmask.height == simg.smooth.rows()) &&
        "Targe Image: dimension does not fit dimension of supplied forest "
        "mask");
    bool m_debug = false;
    // Match
    std::vector<ndb::Correspondence> corr =
        depthPriorFast(simg, timg, forestmask, settings);

    return corr;
}

/**
 * @brief                   Returns support (set of x,y coordinates and
 * disparity) of a pair of images that have been rectified.
 *
 * @@param simg             source image (assumed to be the left image)
 * @param timg              target image (assumed to be the right image)
 * @param forestmask        forest mask, provided by readForest method
 * @param InferenceSettings inference settings struct
 *                          In practice, values between 5...20 produce good
 * results.
 *
 * @return                  Set of supports (x,y,d) with x,y the coordinate
 * of a point in the left image and d the disparity.
 */
std::vector<ndb::Support> Forest::rectifiedMatch(PreprocessedImage& simg,
                                         PreprocessedImage& timg,
                                         FilterMask& forestmask,
                                         InferenceSettings settings) {
    // Do matching
    std::vector<ndb::Correspondence> corr =
        stereoMatch(simg, timg, forestmask, settings);
    // Filter epipolar matches
    std::vector<ndb::Support> supp;
    for (auto& e : corr) {
        // epipolar constraint
        if (std::abs(e.srcPt.y - e.tarPt.y) <= settings.verticalTolerance_
            // disparity filter
            && std::abs(e.srcPt.x - e.tarPt.x) <= settings.dispHigh_)
            supp.push_back(
                ndb::Support(e.srcPt.x, e.srcPt.y, e.srcPt.x - e.tarPt.x));
    }
    return supp;
}

/**
 * @brief Reads text-based forest format and returns a mask for a given
 * image size.
 *
 * @param path    Path to the file that contains the forest.
 * @param width   16-Byte aligned width of the image in pixels
 * @param height  height of the image in pixels
 *
 * @return
 */
FilterMask Forest::readForest(std::string path, int width, int height) {
    std::ifstream ff(path);

    std::vector<int32_t> fastmask;
    std::vector<int> taus;
    if (ff.fail()) {
        cout << "Error opening forest file" << endl;
        return FilterMask(fastmask, width, height, 0);
    }
    int numNonZeroTau = 0;
    int numFerns;
    int type;
    ff >> numFerns;
    cout << "number of ferns:" << numFerns << endl;
    for (int i = 0; i < numFerns; i++) {
        int fernID, numTests;
        std::string fernScale;
        ff >> fernID >> fernScale >> numTests;
        for (int j = 0; j < numTests; j++) {
            int levelID, ix, iy, jx, jy, tau;
            ff >> levelID >> ix >> iy >> jx >> jy >> tau;
            // Limit mask size to 32 binary tests
            if (fastmask.size() < 64 && taus.size() < 32) {
                fastmask.push_back(ix + iy * width);
                fastmask.push_back(jx + jy * width);
                taus.push_back(tau);
            } else {
                cout << "Note: A maximum of 32 fern features are allowed, "
                        "discarding "
                        "remainder of forest."
                     << endl;
            }
            if (tau != 0) numNonZeroTau++;
        }
    }
    if (numNonZeroTau == 0) {
        type = 0;  // We have a zero forest (all tau=0)
        FilterMask fm(fastmask, width, height, type);
        return fm;
    } else {
        type = 1;  // We have a tau forest (some tau!=0)
        FilterMask fm(fastmask, taus, width, height, type);
        return fm;
    }
}

}  // namespace inference
}
