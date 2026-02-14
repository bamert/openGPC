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
#ifndef _GPC_fern
#define _GPC_fern
#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

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
    OptimizerSettings(int taulo,
                      int tauhi,
                      int numResamples,
                      bool onlyScoreNonSplitSamples,
                      double w1)
        : taulo_(taulo),
          tauhi_(tauhi),
          numResamples_(numResamples),
          onlyScoreNonSplitSamples_(onlyScoreNonSplitSamples),
          w1_(w1) {};
    OptimizerSettings() {}
};
struct TauOptimizerSettings : public OptimizerSettings {
    TauOptimizerSettings(int taulo,
                         int tauhi,
                         int numResamples,
                         bool onlyScoreNonSplitSamples,
                         double w1)
        : OptimizerSettings(
              taulo, tauhi, numResamples, onlyScoreNonSplitSamples, w1) {}
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
    TauOptimizerSettings& onlyScoreNonSplitSamples(
        bool onlyScoreNonSplitSamples) {
        this->onlyScoreNonSplitSamples_ = onlyScoreNonSplitSamples;
        return *this;
    }
};
struct ZeroOptimizerSettings : public OptimizerSettings {
    ZeroOptimizerSettings(int numResamples,
                          bool onlyScoreNonSplitSamples,
                          double w1)
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
    ZeroOptimizerSettings& onlyScoreNonSplitSamples(
        bool onlyScoreNonSplitSamples) {
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
 * @param w1        Weight for weighted harmonic mean between precision and
 * recall
 *
 * @return
 */
OptimizerSettings TauOptimizer(int taulo,
                               int tauhi,
                               int numResamples,
                               bool onlyScoreNonSplitSamples,
                               double w1);
/**
 * @brief Optimzer setting factory for a zero fern
 *
 * @param numResamples number of resamplings for each hyperplane
 * @param onlyScoreNonSplitSamples if true, only score samples
 *                  that have not been true positive in previous fern leves
 * @param w1        Weight for weighted harmonic mean between precision and
 * recall
 *
 * @return
 */
OptimizerSettings ZeroOptimizer(int numResamples,
                                bool onlyScoreNonSplitSamples,
                                double w1) ;
struct FernSettings {
    const int maxDepth;
    const int scale;
    FernSettings(int maxDepth, int scale) : maxDepth(maxDepth), scale(scale) {};
};

/**
 * @brief      The Tau Fern uses a threshold, i.e. the learned tests are
 * \phi(x;i,j,w) := sign(x(i)-x(j)-w)
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
    void evalSplit(std::vector<GPCTriplet_t>& data,
                   std::vector<SplitParams_t>& params,
                   FernSettings fernsetting,
                   OptimizerSettings optsetting,
                   int scoreUntilLevel,
                   splitStats& s);
     /**
     * @brief      Mark those samples in the set as "split" if they have been
     *             correctly classified(ref=pos and pos!=neg) with the parameter
     * set in params
     *
     * @param[in]  data       The dataset
     * @param      params     The parameters
     * @param[in]  numParams  The number parameters
     */
    void markSplitSamples(std::vector<GPCTriplet_t>& data,
                          std::vector<SplitParams_t>& params,
                          int numParams) ;
    /**
     * @brief Reset the mark on the training samples on whether they have been
     * split correctly or not Since we do not operate on copies of the training
     * set for each fern, this is required.
     *
     * @param data
     */
    void resetMarkOnSamples(std::vector<GPCTriplet_t>& data);
   
    /**
     * @brief Train a fern given a set of training data and some optimizer
     * settings
     *
     * @param trainingSamples The training samples
     * @param optsetting      the optimizer settings
     */
    void train(std::vector<GPCTriplet_t>& trainingSamples,
               OptimizerSettings optsetting) ;
   
    /**
     * @brief      Returns the decision of the first five levels of the ferns
     *
     * @return     The parameters.
     */
    std::vector<SplitParams_t> getParameters();

    /**
     * @brief Return the scale that this fern uses
     *
     * @return The scale
     */
    int getScale();

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
inline std::vector<Fern> FernFactory(int num_S, int num_M, int num_L, int maxDepth) {
    std::vector<Fern> ferns;
    for (int i = 0; i < num_S; i++)
        ferns.push_back(Fern(FernSettings(maxDepth, 2)));
    for (int i = 0; i < num_M; i++)
        ferns.push_back(Fern(FernSettings(maxDepth, 1)));
    for (int i = 0; i < num_L; i++)
        ferns.push_back(Fern(FernSettings(maxDepth, 0)));
    return ferns;
}
}  // namespace training
}  // namespace gpc
#endif
