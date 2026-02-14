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
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "gpc/Feature.hpp"
#include "gpc/Fern.hpp"

using namespace std;
namespace gpc {
namespace training {
OptimizerSettings TauOptimizer(int taulo,
                               int tauhi,
                               int numResamples,
                               bool onlyScoreNonSplitSamples,
                               double w1) {
    return OptimizerSettings(
        taulo, tauhi, numResamples, onlyScoreNonSplitSamples, w1);
}
OptimizerSettings ZeroOptimizer(int numResamples,
                                bool onlyScoreNonSplitSamples,
                                double w1) {
    return OptimizerSettings(0, 1, numResamples, onlyScoreNonSplitSamples, w1);
}
void Fern::evalSplit(std::vector<GPCTriplet_t>& data,
               std::vector<SplitParams_t>& params,
               FernSettings fernsetting,
               OptimizerSettings optsetting,
               int scoreUntilLevel,
               splitStats& s) {
    s.tp = 0;
    s.fn = 0;
    s.fp = 0;
    s.prec = 0.;
    s.rec = 0.;
    s.hmean = 0.;
    s.convcomb = 0.;
    s.tot = 0;
    for (auto& triplet : data) {
        uint64_t ref = 0, pos = 0, neg = 0;
        // Score the first scoreUntilLevel levels of a given fern
        for (int i = 0; i < scoreUntilLevel + 1; i++) {
            ref <<= 1;
            pos <<= 1;
            neg <<= 1;
            bool refDec, posDec, negDec;

            // Decisions need to be added into a codeword
            Feature.getDecisions(
                refDec, posDec, negDec, params[i], triplet);
            if (refDec) ref++;
            if (posDec) pos++;
            if (negDec) neg++;
        }
        // Only count those that haven't been true positives yet
        // Ignore samples previously classified as True positive
        if (!(triplet.pos.split == true && triplet.neg.split == true)) {
            s.tot++;
            // Decide which are equal (i.e. set the split indicators)
            if (ref == pos) {      // 110(TP), 111, 001(TP), 000
                if (ref != neg) {  // 110 (TP), 001(TP)
                    s.tp++;
                } else {  // 111(FN), 000(FN)
                    s.fn++;
                }
            } else {               // 100, 101, 011, 010
                if (ref != neg) {  // 100(FN), 011(FN) FN
                    s.fn++;
                } else {  //  101(FP), 010(FP)
                    s.fp++;
                }
            }
        }
    }

    // Compute statistics of this split
    double w2 = 1. - optsetting.w1_;
    s.prec = ((s.tp + s.fp) == 0) ? 0. : double(s.tp) / (s.tp + s.fp);
    s.rec = ((s.tp + s.fn) == 0) ? 0. : double(s.tp) / (s.tp + s.fn);

    s.hmean = (s.prec + s.rec == 0.)
                  ? 0.
                  : s.prec * s.rec / ((1. - w2) * s.prec + w2 * s.rec);
    s.convcomb = (1. - w2) * s.prec + w2 * s.rec;
}
void Fern::markSplitSamples(std::vector<GPCTriplet_t>& data,
                      std::vector<SplitParams_t>& params,
                      int numParams) {
    for (auto& triplet : data) {
        // Evaluate triplet on all given parameters
        uint64_t ref = 0, pos = 0, neg = 0;
        for (int i = 0; i < numParams; i++) {
            ref <<= 1;  // shift by one
            pos <<= 1;  // shift by one
            neg <<= 1;  // shift by one
            bool refDec, posDec, negDec;

            Feature.getDecisions(
                refDec, posDec, negDec, params[i], triplet);
            if (refDec) ref++;
            if (posDec) pos++;
            if (negDec) neg++;
        }
        if (ref == pos) triplet.pos.split = true;
        if (ref != neg) triplet.neg.split = true;
    }
}
void Fern::resetMarkOnSamples(std::vector<GPCTriplet_t>& data) {
    for (auto& triplet : data) {
        triplet.pos.split = false;
        triplet.neg.split = false;
    }
}

void Fern::train(std::vector<GPCTriplet_t>& trainingSamples,
           OptimizerSettings optsetting) {
    splitStats stats;
    float maxScore = 0.f;
    SplitParams_t bestParams;

    fernparams.resize(fernsettings.maxDepth);

    cout << setw(7) << "Level" << setw(10) << "Prec" << setw(10) << "Rec"
         << setw(10) << "Har" << setw(8) << "Tot" << setw(8) << "TP"
         << setw(8) << "FP" << setw(8) << "FN" << setw(6) << "scale"
         << setw(5) << "tau" << setw(5) << "i" << setw(5) << "j" << endl;
    if (optsetting.onlyScoreNonSplitSamples_)
        resetMarkOnSamples(trainingSamples);
    for (int level = 0; level < fernsettings.maxDepth; level++) {
        maxScore = 0.f;
        for (int k = 0; k < optsetting.numResamples_; k++) {
            // Samples a hyperplane in the requested scale
            Feature.sampleHyperplane(fernsettings.scale, fernparams[level]);
            // Iterates over a small range of tau (intercept)
            for (int tau = optsetting.taulo_; tau < optsetting.tauhi_;
                 tau++) {
                fernparams[level].tau = tau;
                // Score hyperplane set we have so far
                evalSplit(trainingSamples,
                          fernparams,
                          fernsettings,
                          optsetting,
                          level,
                          stats);
                // If score exceeds previously best, replace paramset
                if (stats.hmean > maxScore) {
                    bestParams = fernparams[level];
                    maxScore = stats.hmean;
                }
            }  // tau loop
        }  // k loop
        // Store best performing parameters
        fernparams[level] = bestParams;

        // Mark samples as split if they were labeled true positive
        if (optsetting.onlyScoreNonSplitSamples_)
            markSplitSamples(trainingSamples, fernparams, level);
        cout << setw(7) << level << setw(10) << stats.prec << setw(10)
             << stats.rec << setw(10) << stats.hmean << setw(8) << stats.tot
             << setw(8) << stats.tp << setw(8) << stats.fp << setw(8)
             << stats.fn << setw(6) << fernsettings.scale << setw(5)
             << fernparams[level].tau << setw(5) << fernparams[level].i
             << setw(5) << fernparams[level].j << endl;
    }  // level loop
}  // train

std::vector<Fern::SplitParams_t> Fern::getParameters() { return fernparams; }

int Fern::getScale() { return fernsettings.scale; }



}  // namespace training
}  // namespace gpc
