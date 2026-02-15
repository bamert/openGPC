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
#include <algorithm>
#include <cmath>  //for log2
#include <fstream>
#include <gpc/buffer.hpp>
#include <gpc/filter.hpp>
#include <gpc/kernels/box.hpp>
#include <gpc/Feature.hpp>
#include <iostream>
#include <iterator>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace std;

namespace gpc {
namespace training {
void Feature::getDecisions(bool& ref,
                         bool& pos,
                         bool& neg,
                         params& params,
                         const GPCPatchTriplet& trip) {
    ref =
        ((int)trip.ref.feature(params.i) - (int)trip.ref.feature(params.j) <
         params.tau);
    pos =
        ((int)trip.pos.feature(params.i) - (int)trip.pos.feature(params.j) <
         params.tau);
    neg =
        ((int)trip.neg.feature(params.i) - (int)trip.neg.feature(params.j) <
         params.tau);
}

Feature::Feature() {
    std::random_device rd2;
    rng = std::mt19937(rd2());
    randIJ7 = std::uniform_int_distribution<int>(0, 48);
    randIJ17 = std::uniform_int_distribution<int>(0, 17 * 17 - 1);
    randIJ27 = std::uniform_int_distribution<int>(0, 27 * 27 - 1);

    randTAU = std::uniform_int_distribution<int>(-15, 15);
}
void Feature::sampleHyperplane(int scale, params& params) {
    if (scale == 2) {
        params.i = params.j;  // s.t. they regenerate each iteration
        while (params.i == params.j) {  // i and j need to be distinct
            int i = randIJ7(rng);
            int j = randIJ7(rng);
            params.ix = i % 7 - 3;
            params.iy = i / 7 - 3;
            params.jx = j % 7 - 3;
            params.jy = j / 7 - 3;

            params.i = 280 + (params.ix + 3) + 27 * (params.iy + 3);
            params.j = 280 + (params.jx + 3) + 27 * (params.jy + 3);
        }
    } else if (scale == 1) {
        params.i = params.j;  // s.t. they regenerate each iteration
        while (params.i == params.j) {  // i and j need to be distinct
            int i = randIJ17(rng);
            int j = randIJ17(rng);
            params.ix = i % 17 - 8;
            params.iy = i / 17 - 8;
            params.jx = j % 17 - 8;
            params.jy = j / 17 - 8;

            params.i = 140 + (params.ix + 8) + 27 * (params.iy + 8);
            params.j = 140 + (params.jx + 8) + 27 * (params.jy + 8);
        }
    } else if (scale == 0) {
        params.i = params.j;  // s.t. they regenerate each iteration
        while (params.i == params.j) {  // i and j need to be distinct
            params.i = randIJ27(rng);
            params.j = randIJ27(rng);
            params.ix = params.i % 27 - 13;
            params.iy = params.i / 27 - 13;
            params.jx = params.j % 27 - 13;
            params.jy = params.j / 27 - 13;

            params.i = (params.ix + 13) + 27 * (params.iy + 13);
            params.j = (params.jx + 13) + 27 * (params.jy + 13);
        }
    }
    params.tau = randTAU(rng);
}
void Feature::extractAllTriplets(ndb::Buffer<uint8_t>& bwL,
                        ndb::Buffer<uint8_t>& bwR,
                        std::vector<ndb::Point>& ref,
                        std::vector<ndb::Point>& pos,
                        std::vector<ndb::Point>& neg,
                        std::vector<GPCPatchTriplet>& triplets) {
    ndb::Buffer<uint8_t> LL(bwL.rows(), bwL.cols());
    LL.width = bwL.width;
    ndb::box(bwL.data(), LL.data(), bwL.cols(), bwL.rows(), 1);
    LL.clearBoundary();

    ndb::Buffer<uint8_t> RR(bwL.rows(), bwL.cols());
    RR.width = bwR.width;
    ndb::box(bwR.data(), RR.data(), bwR.cols(), bwR.rows(), 1);
    RR.clearBoundary();

    auto f = [=](ndb::Point& kp) {
        if (kp.x > 20 && kp.y > 20 && kp.x < bwL.cols() - 20 &&
            kp.y < bwL.rows() - 20)
            return false;
        else
            return true;
    };

    for (std::vector<ndb::Point>::size_type i = 0; i != ref.size(); i++) {
        if (!f(ref[i]) && !f(pos[i]) && !f(neg[i])) {
            // Get all descriptors:
            GPCPatchTriplet newPatch;

            // Reference patch
            //====================================
            newPatch.ref.x = ref[i].x;
            newPatch.ref.y = ref[i].y;

            LL.getPatch(newPatch.ref.feature, ref[i].x, ref[i].y, 27);

            // Extract a positive match in the right image
            //====================================
            newPatch.pos.x = pos[i].x;
            newPatch.pos.y = pos[i].y;

            RR.getPatch(newPatch.pos.feature, pos[i].x, pos[i].y, 27);

            // Extract negative patch
            //====================================
            newPatch.neg.x = neg[i].x;
            newPatch.neg.y = neg[i].y;

            RR.getPatch(newPatch.neg.feature, neg[i].x, neg[i].y, 27);

            triplets.push_back(std::move(newPatch));
        }
    }
}

void Feature::storeAllTriplets(std::vector<GPCPatchTriplet>& data,
                      std::string path) {
    ofstream fout;
    fout.open(path, ios::binary | ios::out);
    for (auto& triplet : data) {
        fout.write((char*)triplet.ref.feature.data(), 27 * 27);
        fout.write((char*)triplet.pos.feature.data(), 27 * 27);
        fout.write((char*)triplet.neg.feature.data(), 27 * 27);
    }
    fout.close();
}
std::vector<Feature::GPCPatchTriplet> Feature::loadAllTriplets(std::string path) {
    std::vector<Feature::GPCPatchTriplet> data;
    std::ifstream in(path, std::ifstream::ate | std::ifstream::binary);
    uint32_t filesize = in.tellg();
    if (filesize % ((27 * 27) * 3)) {
        cout << "ERR: File is not a training set of this feature type"
             << endl;
        cout << "FS: " << filesize << endl;
        return data;
    }
    int numSamples = filesize / ((27 * 27) * 3);
    data.resize(numSamples);
    ifstream fin;
    fin.open(path, ios::binary | ios::in);
    for (auto& datum : data) {
        datum.ref.feature.resize(27, 27);
        datum.pos.feature.resize(27, 27);
        datum.neg.feature.resize(27, 27);

        fin.read((char*)datum.ref.feature.data(), 27 * 27);
        fin.read((char*)datum.pos.feature.data(), 27 * 27);
        fin.read((char*)datum.neg.feature.data(), 27 * 27);
    }
    fin.close();
    return data;
}

}  // namespace training
}  // namespace gpc
