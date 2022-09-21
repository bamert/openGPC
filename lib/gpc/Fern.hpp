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
#ifndef _GPC_fern
#define _GPC_fern
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>

#include "gpc/Feature.hpp"

using namespace std;
namespace gpc {
namespace training {

/**
 * @brief Contains the statistics of a given split.
 *        This is printed after each level of a fern is trained
 */
struct splitStats {
    // Precision and recall
    double prec = 0.;
    double rec = 0.;
    // Weighted Harmonic mean of precision and recall,
    // weighted with the w1 parameter, which is
    // set globally for an entire fern in the optimizer settings
    double hmean = 0.;
    // Convex combination of precision and recall
    double convcomb = 0.;
    // True positives(tp), false positives (fp) and false negatives(fn)
    int tp = 0;
    int fp = 0;
    int fn = 0;
    // Total number of patches (tp+fp+fn)
    int tot = 0;
};

struct OptimizerSettings {
    // Weight for the weighted harmonic mean that is used to
    // optimize the fern splits greedily
    double w1_;
    // Number of resamplings of the hyperplane in each split
    // The highest scoring of numResamples splits is chosen
    int numResamples_;
    // The search interval of the intercept in the learned tests.
    // For zeroferns, we have taulo=0,tauhi=1, which forces the
    // intercept to be 0.
    int taulo_;
    int tauhi_;
    // If true, only those training samples are used for scoring
    // that have not been split successfully in previous
    // fern levels. (I.e. they have not been a true positive yet)
    bool onlyScoreNonSplitSamples_;
    OptimizerSettings(
        int taulo, int tauhi, int numResamples, bool onlyScoreNonSplitSamples, double w1)
        : taulo_(taulo),
          tauhi_(tauhi),
          numResamples_(numResamples),
          onlyScoreNonSplitSamples_(onlyScoreNonSplitSamples),
          w1_(w1){};
    OptimizerSettings() {}
};
struct TauOptimizerSettings : public OptimizerSettings {
    TauOptimizerSettings(
        int taulo, int tauhi, int numResamples, bool onlyScoreNonSplitSamples, double w1)
        : OptimizerSettings(taulo, tauhi, numResamples, onlyScoreNonSplitSamples, w1) {}
    TauOptimizerSettings() : OptimizerSettings() {}

    TauOptimizerSettings& builder(void) { return *this; }
    TauOptimizerSettings& w1(double w1) {
        this->w1_ = w1;
        return *this;
    }
    TauOptimizerSettings& numResamples(double numResamples) {
        this->numResamples_ = numResamples;
        return *this;
    }
    TauOptimizerSettings& taulo(double taulo) {
        this->taulo_ = taulo;
        return *this;
    }
    TauOptimizerSettings& tauhi(int tauhi) {
        this->tauhi_ = tauhi;
        return *this;
    }
    TauOptimizerSettings& onlyScoreNonSplitSamples(bool onlyScoreNonSplitSamples) {
        this->onlyScoreNonSplitSamples_ = onlyScoreNonSplitSamples;
        return *this;
    }
};
struct ZeroOptimizerSettings : public OptimizerSettings {
    ZeroOptimizerSettings(int numResamples, bool onlyScoreNonSplitSamples, double w1)
        : OptimizerSettings(0, 1, numResamples, onlyScoreNonSplitSamples, w1) {}
    ZeroOptimizerSettings() : OptimizerSettings() {}

    ZeroOptimizerSettings& builder(void) { return *this; }
    ZeroOptimizerSettings& w1(double w1) {
        this->w1_ = w1;
        return *this;
    }
    ZeroOptimizerSettings& numResamples(double numResamples) {
        this->numResamples_ = numResamples;
        return *this;
    }
    ZeroOptimizerSettings& onlyScoreNonSplitSamples(bool onlyScoreNonSplitSamples) {
        this->onlyScoreNonSplitSamples_ = onlyScoreNonSplitSamples;
        return *this;
    }
};
/**
 * @brief Optimzer setting factory for a tau fern
 *
 * @param taulo     lower end of search range for the learned intercept
 * @param tauhi     upper end of search range for the learned intercept
 * @param numResamples number of resamplings for each hyperplane
 * @param onlyScoreNonSplitSamples if true, only score samples
 *                  that have not been true positive in previous fern leves
 * @param w1        Weight for weighted harmonic mean between precision and recall
 *
 * @return
 */
OptimizerSettings TauOptimizer(
    int taulo, int tauhi, int numResamples, bool onlyScoreNonSplitSamples, double w1) {
    return OptimizerSettings(taulo, tauhi, numResamples, onlyScoreNonSplitSamples, w1);
}
/**
 * @brief Optimzer setting factory for a zero fern
 *
 * @param numResamples number of resamplings for each hyperplane
 * @param onlyScoreNonSplitSamples if true, only score samples
 *                  that have not been true positive in previous fern leves
 * @param w1        Weight for weighted harmonic mean between precision and recall
 *
 * @return
 */
OptimizerSettings ZeroOptimizer(
    int numResamples, bool onlyScoreNonSplitSamples, double w1) {
    return OptimizerSettings(0, 1, numResamples, onlyScoreNonSplitSamples, w1);
}
struct FernSettings {
    const int maxDepth;
    const int scale;
    FernSettings(int maxDepth, int scale) : maxDepth(maxDepth), scale(scale){};
};

/**
 * @brief      The Tau Fern uses a threshold, i.e. the learned tests are \phi(x;i,j,w) :=
 * sign(x(i)-x(j)-w)
 * @tparam     Feature_t  The feature to be used in this fern (SL, EFIDG, EFIDS)
 */
class Fern {
private:
    typedef typename gpc::training::Feature Feature_t;
    // Keeps the type the triplets of the chosen Feature F
    typedef typename Feature_t::GPCPatchTriplet GPCTriplet_t;

    typedef typename Feature_t::params SplitParams_t;
    // Instantiation of feature for the dot product method
    Feature_t Feature;

    // Keeps learned parameters of the fern
    std::vector<SplitParams_t> fernparams;

    // Settings(hyperparameters) for this fern
    FernSettings fernsettings;

public:
    Fern(FernSettings fernsettings) : fernsettings(fernsettings) {}
    /**
     * @brief      Evaluates a parameter set for a fern for the range
     *             of levels [0, scoreUntillevel] in the fern.
     *
     * @param[in]  data             The dataset
     * @param      params           The parameters (set of pixels i,j)
     * @param[in]  scoreUntilLevel  The number of sets of (i,j)
     *                       to score as we grow further down the fern.
     *
     */
    void evalSplit(
        std::vector<GPCTriplet_t>& data,
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
                Feature.getDecisions(refDec, posDec, negDec, params[i], triplet);
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
    /**
     * @brief      Mark those samples in the set as "split" if they have been
     *             correctly classified(ref=pos and pos!=neg) with the parameter set in
     * params
     *
     * @param[in]  data       The dataset
     * @param      params     The parameters
     * @param[in]  numParams  The number parameters
     */
    void markSplitSamples(
        std::vector<GPCTriplet_t>& data,
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

                Feature.getDecisions(refDec, posDec, negDec, params[i], triplet);
                if (refDec) ref++;
                if (posDec) pos++;
                if (negDec) neg++;
            }
            if (ref == pos) triplet.pos.split = true;
            if (ref != neg) triplet.neg.split = true;
        }
    }
    /**
     * @brief Reset the mark on the training samples on whether they have been split
     * correctly or not Since we do not operate on copies of the training set for each
     * fern, this is required.
     *
     * @param data
     */
    void resetMarkOnSamples(std::vector<GPCTriplet_t>& data) {
        for (auto& triplet : data) {
            triplet.pos.split = false;
            triplet.neg.split = false;
        }
    }

    /**
     * @brief Train a fern given a set of training data and some optimizer settings
     *
     * @param trainingSamples The training samples
     * @param optsetting      the optimizer settings
     */
    void train(std::vector<GPCTriplet_t>& trainingSamples, OptimizerSettings optsetting) {
        splitStats stats;
        float maxScore = 0.f;
        SplitParams_t bestParams;

        fernparams.resize(fernsettings.maxDepth);

        cout << setw(7) << "Level" << setw(10) << "Prec" << setw(10) << "Rec" << setw(10)
             << "Har" << setw(8) << "Tot" << setw(8) << "TP" << setw(8) << "FP" << setw(8)
             << "FN" << setw(6) << "scale" << setw(5) << "tau" << setw(5) << "i"
             << setw(5) << "j" << endl;
        if (optsetting.onlyScoreNonSplitSamples_) resetMarkOnSamples(trainingSamples);
        for (int level = 0; level < fernsettings.maxDepth; level++) {
            maxScore = 0.f;
            for (int k = 0; k < optsetting.numResamples_; k++) {
                // Samples a hyperplane in the requested scale
                Feature.sampleHyperplane(fernsettings.scale, fernparams[level]);
                // Iterates over a small range of tau (intercept)
                for (int tau = optsetting.taulo_; tau < optsetting.tauhi_; tau++) {
                    fernparams[level].tau = tau;
                    // Score hyperplane set we have so far
                    evalSplit(
                        trainingSamples, fernparams, fernsettings, optsetting, level,
                        stats);
                    // If score exceeds previously best, replace paramset
                    if (stats.hmean > maxScore) {
                        bestParams = fernparams[level];
                        maxScore = stats.hmean;
                    }
                }  // tau loop
            }      // k loop
            // Store best performing parameters
            fernparams[level] = bestParams;

            // Mark samples as split if they were labeled true positive
            if (optsetting.onlyScoreNonSplitSamples_)
                markSplitSamples(trainingSamples, fernparams, level);
            cout << setw(7) << level << setw(10) << stats.prec << setw(10) << stats.rec
                 << setw(10) << stats.hmean << setw(8) << stats.tot << setw(8) << stats.tp
                 << setw(8) << stats.fp << setw(8) << stats.fn << setw(6)
                 << fernsettings.scale << setw(5) << fernparams[level].tau << setw(5)
                 << fernparams[level].i << setw(5) << fernparams[level].j << endl;
        }  // level loop
    }      // train

    /**
     * @brief      Returns the decision of the first five levels of the ferns
     *
     * @return     The parameters.
     */
    std::vector<SplitParams_t> getParameters() { return fernparams; }

    /**
     * @brief Return the scale that this fern uses
     *
     * @return The scale
     */
    int getScale() { return fernsettings.scale; }

};  // Fern

/**
 * @brief Fern factory. Returns a set of ferns
 *
 * @param num_S   Number of  7 x 7 ferns
 * @param num_M   Number of 17 x 17 fern
 * @param num_L   Number of 27 x 27 ferns
 * @param maxDepth)
 *
 * @return
 */
std::vector<Fern> FernFactory(int num_S, int num_M, int num_L, int maxDepth) {
    std::vector<Fern> ferns;
    for (int i = 0; i < num_S; i++) ferns.push_back(Fern(FernSettings(maxDepth, 2)));
    for (int i = 0; i < num_M; i++) ferns.push_back(Fern(FernSettings(maxDepth, 1)));
    for (int i = 0; i < num_L; i++) ferns.push_back(Fern(FernSettings(maxDepth, 0)));
    return ferns;
}
}  // namespace training
}  // namespace gpc
#endif
