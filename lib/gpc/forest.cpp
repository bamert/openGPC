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
        std::vector<ndb::Correspondence> corr =
            findCorrespondences(statesSrc, statesTar);
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
                       img.rows(),
                       settings.numThreads_);
    } else {
        ndb::gpcFilterTau(img.data(),
                          grad.data(),
                          gpcstates.data(),
                          fastmask.mask,
                          fastmask.tau,
                          idx,
                          img.cols(),
                          img.rows(),
                          settings.numThreads_);
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
    gpc::inference::time_point t0 = gpc::inference::sysTick();
    ndb::box(img.data(),
             smooth.data(),
             img.cols(),
             img.rows(),
             settings.numThreads_);
    gpc::inference::time_point t1 = gpc::inference::sysTick();
    cout << "box: " << gpc::inference::tickToMs(t1, t0) << " ms" << endl;
    smooth.clearBoundary();
    ndb::Buffer<uint8_t> grad(img.rows(), img.cols());
    grad.width = img.width;
    ndb::Buffer<int> maskTmp;
    ndb::sobel(img.data(),
               grad.data(),
               img.cols(),
               img.rows(),
               settings.gradientThreshold_,
               settings.numThreads_);

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
    std::chrono::high_resolution_clock::time_point t0, t1;
    // Match
    std::vector<ndb::Correspondence> corr =
        depthPriorFast(simg, timg, forestmask, settings);
    t1 = sysTick();

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
