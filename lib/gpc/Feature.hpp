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
#ifndef _GPC_feature
#define _GPC_feature

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>  //for log2
#include <fstream>
#include <gpc/buffer.hpp>
#include <iostream>
#include <iterator>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace std;

namespace gpc {
namespace training {
class Feature {
   private:
    std::mt19937 rng;
    std::uniform_int_distribution<int> randIJ7, randIJ17, randIJ27, randTAU;

   public:
    // Type of data inside a descriptor.

    struct GPCDescriptor {
        ndb::Buffer<uint8_t> feature;
        int x, y;
        bool split = false;  // indicates whether this sample has been split
                             // from the reference in training.
        bool le = false;     // marks a patch as low energy.
    };

    /**
     * @brief     Contains a triplet of image patches
     */
    struct GPCPatchTriplet {
        GPCDescriptor ref;
        GPCDescriptor pos;
        GPCDescriptor neg;
    };

    /**
     * @brief     Contains the parameters for a single level inside a fern
     */
    struct params {
        int i = 0, j = 0;
        int ix = 0, iy = 0;
        int jx = 0, jy = 0;
        int tau = 0;  // threshold for sign(i-j-tau)
    };

    /**
     * @brief      Gets the left / right decisions for each patch in a given
     * triplet of image patches.
     *
     * @param      ref     The reference decision
     * @param      pos     The positive decision
     * @param      neg     The negative decision
     * @param      params  The parameters for this split
     * @param[in]  trip    The triplet
     */
    void getDecisions(bool& ref,
                             bool& pos,
                             bool& neg,
                             params& params,
                             const GPCPatchTriplet& trip);

    Feature();
    /**
     * @brief Returns a random hyperplane within a 27 x 27
     *        pixel-sized patch. depending on the scale
     *        parameter, the coordinates are selected
     *        s.t. they lie inside a 7x7, 17x17 or within
     *        the entire 27x27 pixels patch region
     *
     * @param scale Determines which patch size is used
     * @param params returns the parameters
     */
    void sampleHyperplane(int scale, params& params);
    /**
     * @brief      Gets all descriptors (triplets) for an image pair for
     * training given the three keypoint vectors.
     *
     * @param      bwL       The bw l
     * @param      bwR       The bw r
     * @param[in]  ref       reference patches coordinates
     * @param[in]  pos       positive patch coordinates
     * @param[in]  neg       negative patch coordinates
     * @param      triplets  The extracted patch triplets
     *
     */
    void extractAllTriplets(ndb::Buffer<uint8_t>& bwL,
                            ndb::Buffer<uint8_t>& bwR,
                            std::vector<ndb::Point>& ref,
                            std::vector<ndb::Point>& pos,
                            std::vector<ndb::Point>& neg,
                            std::vector<GPCPatchTriplet>& triplets);
    /**
     * @brief Store a vector of triplets of training data to file
     *
     * @param data The triplet vector
     * @param path The path where we'd like to store the training data
     *             in binary form.
     */
    void storeAllTriplets(std::vector<GPCPatchTriplet>& data,
                          std::string path);
    /**
     * @brief Read triplets of training data from a binary file
     *        written by the storeAllTriplets method.
     *
     * @param path path to the binary file
     *
     * @return The training set
     */
    std::vector<GPCPatchTriplet> loadAllTriplets(std::string path);
};  // Feature
}  // namespace training
}  // namespace gpc
#endif
