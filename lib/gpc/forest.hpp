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
#ifndef _GPC_inference
#define _GPC_inference
#include <Eigen/Dense>
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
#include "gpc/hashmatch.hpp"

/**
 * @brief      The inference class of the GPC forest
 *
 */
namespace gpc {
namespace inference {
typedef typename std::chrono::high_resolution_clock::time_point time_point;
inline std::chrono::high_resolution_clock::time_point sysTick() {
    return std::chrono::high_resolution_clock::now();
}
inline float tickToMs(std::chrono::high_resolution_clock::time_point t0,
               std::chrono::high_resolution_clock::time_point t1) {
    return std::abs(
        1000. *
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
            .count());
}
struct InferenceSettings {
    // Threshold to be used for edge detection. Can be 0...255.
    // In practice, values between 5...20 produce good results.  uint8_t
    // gradientThreshold;
    uint8_t gradientThreshold_ = 10;
    // upper absolute limit for disparity in pixels. The lower (implied) limit
    // is
    // 0
    int dispHigh_ = 128;
    // vertical deviation tolerance in pixels for corresponding features in
    // rectified stereo images.
    int verticalTolerance_ = 1;
    // Whether to use epipolar mode on matching or not.
    bool epipolarMode_ = false;
    // Use hashtable to match extracted descriptors. Usually only faster with a
    // large number of descriptors (> 100k) or when using multiple threads. Note
    // that the hashtable method does not return a slightly reduced amount of
    // matches as a result of the hash table implementation (small bucket size)
    // if false, the descriptors are sorted and matched by iterating
    // alternatingly through both sets.
    bool useHashtable_ = false;

    // Number of threads to use for inference
    int numThreads_ = 1;

    // Default contructor defaults to using a single thread
    InferenceSettings(uint8_t gradientThreshold,
                      int dispHigh,
                      int verticalTolerance,
                      bool epipolarMode,
                      bool useHashtable,
                      int numThreads)
        : gradientThreshold_(gradientThreshold),
          dispHigh_(dispHigh),
          verticalTolerance_(verticalTolerance),
          epipolarMode_(epipolarMode),
          useHashtable_(useHashtable),
          numThreads_(numThreads) {}

    InferenceSettings() {}
    InferenceSettings& builder(void) { return *this; }
    InferenceSettings& gradientThreshold(uint8_t gradientThreshold) {
        this->gradientThreshold_ = gradientThreshold;
        return *this;
    }
    InferenceSettings& dispHigh(int dispHigh) {
        this->dispHigh_ = dispHigh;
        return *this;
    }
    InferenceSettings& verticalTolerance(int verticalTolerance) {
        this->verticalTolerance_ = verticalTolerance;
        return *this;
    }
    InferenceSettings& epipolarMode(bool epipolarMode) {
        this->epipolarMode_ = epipolarMode;
        return *this;
    }
    InferenceSettings& useHashtable(bool useHashtable) {
        this->useHashtable_ = useHashtable;
        return *this;
    }
    InferenceSettings& numThreads(int numThreads) {
        if (numThreads > std::thread::hardware_concurrency())
            this->numThreads_ = std::thread::hardware_concurrency();
        else
            this->numThreads_ = numThreads;
        return *this;
    }
};
/**
 * @brief FilterMask object that is returned by the forest reader
 */
struct FilterMask {
    std::vector<int32_t> mask;
    std::vector<int> tau;
    int width;
    int height;
    int type;
    FilterMask(std::vector<int32_t> mask, int width, int height, int type) {
        this->mask = mask;
        this->width = width;
        this->height = height;
        this->type = type;
    }
    FilterMask(std::vector<int32_t> mask,
               std::vector<int> tau,
               int width,
               int height,
               int type) {
        this->mask = mask;
        this->tau = tau;
        this->width = width;
        this->height = height;
        this->type = type;
    }
};
struct PreprocessedImage {
    ndb::Buffer<uint8_t> smooth;
    ndb::Buffer<uint8_t> grad;
    std::vector<int> mask;
    PreprocessedImage(ndb::Buffer<uint8_t>& smooth,
                      ndb::Buffer<uint8_t>& grad,
                      std::vector<int>& mask)
        : smooth(smooth), grad(grad), mask(mask) {};
};

enum CorrMethod { sorting = 's', hashtable = 'h' };
struct MatchStats {
    double prec, rec, timeProp, timeMatch;
    int numInlier, numStates, numMatches;
};
struct SoAFrame {
    // 256 Buckets to ensure each chunk fits in L2/L3 cache
    std::vector<uint64_t> states[256];
    std::vector<uint32_t> indices[256];
    
    void reserve(size_t total_size) {
        for(int i=0; i<256; ++i) {
            states[i].reserve(total_size / size_t(256 * 1.2));
            indices[i].reserve(total_size / size_t(256 * 1.2));
        }
    }
};
struct SoAFramePersistent {
    // Persistent memory blocks
    std::vector<uint64_t> statesSlab;
    std::vector<uint32_t> indicesSlab;
    
    // Pointers into the slab for each bucket
    uint64_t* bucketStates[256];
    uint32_t* bucketIndices[256];
    uint32_t bucketSizes[256];

    void preallocate(size_t total_size) {
        statesSlab.assign(total_size, 0);
        indicesSlab.assign(total_size, 0);
    }
};
struct StateIdx {
    uint64_t state;
    uint32_t index;
};

struct SoAFramePersistentSingleSlab {
    std::vector<StateIdx> slab; 
    StateIdx* bucketData[256];
    uint32_t bucketSizes[256];

    void preallocate(size_t total_size) {
        slab.assign(total_size, {0, 0});
    }
};

class Forest {
   public:
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
    std::vector<ndb::Correspondence> depthPriorFast(
        PreprocessedImage& src,
        PreprocessedImage& tar,
        FilterMask& fastmask,
        InferenceSettings& settings);
    static std::vector<ndb::Correspondence> findCorrespondences(
        std::vector<ndb::Descriptor>& srcStates,
        std::vector<ndb::Descriptor>& tarStates);
    static std::vector<ndb::Correspondence> findCorrespondencesHashNaive(
        std::vector<ndb::Descriptor>& srcStates,
        std::vector<ndb::Descriptor>& tarStates);
    static std::vector<ndb::Correspondence> findCorrespondencesHash(
        std::vector<ndb::Descriptor>& srcStates,
        std::vector<ndb::Descriptor>& tarStates);

    static std::vector<ndb::Correspondence> findCorrespondencesHashingRadix(
        std::vector<ndb::Descriptor>& srcStates,
        std::vector<ndb::Descriptor>& tarStates);

    static std::vector<ndb::Correspondence> findCorrespondencesTurbo(
        std::vector<ndb::Descriptor>& srcStates,
        std::vector<ndb::Descriptor>& tarStates);


    static std::pair<SoAFrame, SoAFrame> prepareSoAFrames(
    std::vector<ndb::Descriptor>& srcStates,
    std::vector<ndb::Descriptor>& tarStates);

    static void prepareSoAFramesPersistent(
        std::vector<ndb::Descriptor>& srcStates,
        std::vector<ndb::Descriptor>& tarStates,
        SoAFramePersistent& srcFrame, 
        SoAFramePersistent& tarFrame);
static void prepareSoAFramesPersistentSingleSlab(
    std::vector<ndb::Descriptor>& srcStates,
    std::vector<ndb::Descriptor>& tarStates,
    SoAFramePersistentSingleSlab& srcFrame, 
    SoAFramePersistentSingleSlab& tarFrame);


static std::pair<std::vector<uint32_t>, std::vector<uint32_t>> matchPreparedFrames( SoAFrame& src, SoAFrame& tar);
static std::pair<std::vector<uint32_t>, std::vector<uint32_t>> matchPreparedFramesFaster( SoAFrame& src, SoAFrame& tar);

static std::pair<std::vector<uint32_t>, std::vector<uint32_t>> matchParallelRadixPartitioning(
    SoAFrame& src, 
    SoAFrame& tar) ;
static std::pair<std::vector<uint32_t>, std::vector<uint32_t>> matchBlockedBloom(
    SoAFrame& src, 
    SoAFrame& tar) ;
static std::pair<std::vector<uint32_t>, std::vector<uint32_t>> matchAdaptive(
    SoAFrame& src, 
    SoAFrame& tar);
static std::pair<std::vector<uint32_t>, std::vector<uint32_t>> matchAdaptivePersistent(
    SoAFramePersistent& src, 
    SoAFramePersistent& tar);
static std::pair<std::vector<uint32_t>, std::vector<uint32_t>> matchPipelinedBranchless(
    SoAFramePersistent& src, 
    SoAFramePersistent& tar);
static void matchPipelinedBranchlessPreallocate(
    SoAFramePersistent& src, 
    SoAFramePersistent& tar,
    std::vector<uint32_t>& resultSrc,
    std::vector<uint32_t>& resultTar);

/*
static std::pair<std::vector<uint32_t>, std::vector<uint32_t>> matchAdaptiveNeon(
    SoAFrame& src, 
    SoAFrame& tar);
*/
static void matchPipelinedBranchlessPreallocateSingleSlab(
    SoAFramePersistentSingleSlab& src, SoAFramePersistentSingleSlab& tar,
    std::vector<uint32_t>& outS, std::vector<uint32_t>& outT);





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
    std::vector<ndb::Descriptor> evalFastMaskOnSubsetSSE(
        ndb::Buffer<uint8_t>& img,
        ndb::Buffer<uint8_t>& grad,
        std::vector<int>& idx,
        FilterMask& fastmask,
        InferenceSettings& settings);
        
    /**
     * @brief Preprocesses an image. (smooth, binary sobel image and gradient
     * pixel indices)
     *
     * @param img     The raw input image to be preprocessed
     * @param InferenceSettings inference settings struct
     *
     * @return the preprocessed image
     */
    PreprocessedImage preprocessImage(ndb::Buffer<uint8_t>& img,
                                      InferenceSettings settings);
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
    std::vector<ndb::Correspondence> stereoMatch(PreprocessedImage& simg,
                                                 PreprocessedImage& timg,
                                                 FilterMask& forestmask,
                                                 InferenceSettings settings);
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
    std::vector<ndb::Support> rectifiedMatch(PreprocessedImage& simg,
                                             PreprocessedImage& timg,
                                             FilterMask& forestmask,
                                             InferenceSettings settings);
                                            
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
    FilterMask readForest(std::string path, int width, int height);
};  // forest class
}  // namespace inference
}  // namespace gpc

#endif
