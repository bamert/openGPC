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
// Shenlong Wang, Sean Ryan Fanello, Christoph Rhemann, Shahram Izadi, Pushmeet Kohli
// CVPR 2016
// Code Author: Niklaus Bamert (bamertn@ethz.ch)
#ifndef _GPC_training
#define _GPC_training
#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <cstring>
#include <random>
#include <chrono>
#include <Eigen/Dense>
#include <thread>

// GPC includes
#include "gpc/buffer.hpp"
#include "gpc/hashmatch.hpp"
#include "gpc/filter.hpp"
#include "gpc/Fern.hpp"
#include "gpc/Feature.hpp"
#include "gpc/SintelOpticalFlow.hpp"
#include "gpc/SintelStereo.hpp"

namespace gpc {
namespace training {
struct ForestSettings {
    enum FernType { Zero, Tau };
    FernType fernType;
    std::string getFernTypeName() {
        if (fernType == FernType::Zero)
            return "zero";
        else
            return "tau";
    }
    double sampleFraction;
    std::vector<gpc::training::Fern> ferns;
    ForestSettings(std::vector<gpc::training::Fern> ferns, double sampleFraction)
        : ferns(ferns), sampleFraction(sampleFraction) {}
};
std::chrono::high_resolution_clock::time_point sysTick() {
    return std::chrono::high_resolution_clock::now();
}
float tickToMs(
    std::chrono::high_resolution_clock::time_point t0,
    std::chrono::high_resolution_clock::time_point t1) {
    return std::abs(
        1000. *
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count());
}
/**
 * @brief      An ensemble  of GPC trees
 *
 * @tparam     F       the feature type of the ref,pos,neg which will together
 *                     form a triplet
 */
class Forest {
private:
    // Keeps the type the triplets of the chosen Feature F
    typedef typename gpc::training::Feature F;
    typedef typename F::GPCPatchTriplet GPCTriplet_t;
    typedef typename gpc::training::Fern T;
    std::mt19937 rng;
    std::uniform_int_distribution<int> randSample;

public:
    Forest() {}

    /**
     * @brief      Train a forest
     *
     * @param      trainingSamples  The training triplets
     */
    void trainAndExport(
        std::vector<GPCTriplet_t>& trainingSamples,
        gpc::training::ForestSettings forestSettings,
        gpc::training::OptimizerSettings optSettings,
        std::string filename) {
        std::chrono::high_resolution_clock::time_point t0, t1;
        std::default_random_engine gen;
        std::bernoulli_distribution dist(forestSettings.sampleFraction);
        std::random_device rd2;

        if (trainingSamples.size() == 0) {
            cout << "ERR: Training set is empty. Aborting." << endl;
            return;
        }

        rng = std::mt19937(rd2());
        randSample = std::uniform_int_distribution<int>(
            0, int(forestSettings.sampleFraction * trainingSamples.size()) - 1);

        // Draw random sub samples for each tree
        int fernIndex = 1;
        for (auto& fern : forestSettings.ferns) {
            std::vector<GPCTriplet_t> subSample;
            // With replacement
            for (int i = 0;
                 i < int(forestSettings.sampleFraction * trainingSamples.size()); i++) {
                subSample.push_back(trainingSamples[randSample(rng)]);
            }

            // Train on the generated subsample
            cout << "Fern(" << fernIndex++ << "/" << forestSettings.ferns.size()
                 << ") num samples:" << subSample.size();
            cout << endl << std::string(90, '*') << endl;
            t0 = std::chrono::high_resolution_clock::now();
            fern.train(subSample, optSettings);
            t1 = std::chrono::high_resolution_clock::now();
            cout << "done in "
                 << std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
                        .count()
                 << " s" << endl
                 << endl;
        }
        // Store trained forest to file
        cout << "Exporting forest" << endl;
        std::fstream file(filename, std::ofstream::out | std::ofstream::trunc);
        file << forestSettings.ferns.size() << endl;
        int f = 0;
        for (auto& fern : forestSettings.ferns) {
            // params for each fern
            std::vector<gpc::training::Feature::params> fparams = fern.getParameters();
            // std::vector<FernParam> fparams = fern.getParameters();
            int scale = fern.getScale();  // 2: small, 1: medium, 0: large
            file << f << " " << ((scale == 2) ? "s" : ((scale == 1) ? "m" : "l")) << " "
                 << fparams.size() << endl;
            int i = 0;
            for (auto& p : fparams) {
                file << int(i) << " " << int(p.ix) << " " << int(p.iy) << " " << int(p.jx)
                     << " " << int(p.jy) << " " << int(p.tau) << endl;
                i++;
            }
            f++;
        }
        file.close();
    }
};  // Forest class
}  // namespace training
}  // namespace gpc
#endif
