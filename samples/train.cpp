#include <iostream>
#include "gpc/training.hpp"

using namespace std;

int main(int argc, char** argv) {
  std::string datasetPath  = "../../data/SintelOpticalFlow-extracted.bin";
  std::string forestPath = "../../forests/defaultZeroForest.txt"; 

  if(argc == 3){
    datasetPath = argv[1];
    forestPath = argv[2];
  }else{
    cout << "Usage: " << argv[0] << " <extracted dataset path> <forest path>" << endl;
    cout << "Trying defaults:" << endl;
    cout << "Extracted dataset path  : " << datasetPath << endl;
    cout << "Export trained forest to: " << forestPath << endl;
  }

  //The three integer arguments denote the number of ferns to train for
  // 10 random hyperplane samples (pick best)
  // Reuse all training samples to train each level of the fern
  // weight recall and precision equally (0.5) 
  gpc::training::OptimizerSettings zerooptimizer = gpc::training::ZeroOptimizerSettings().builder()
    .numResamples(10)
    .onlyScoreNonSplitSamples(false)
    .w1(0.5);

  // Alternatively use Tau fern, additionally train intercept
  // line search on range (-10, 10) for intercept
  // weight recall and precision equally (0.5) 
  gpc::training::OptimizerSettings tauoptimizer = gpc::training::TauOptimizerSettings().builder()
    .taulo(-10)
    .tauhi(10)
    .numResamples(10)
    .onlyScoreNonSplitSamples(false)
    .w1(0.5);
  // Build forest of 2 small, 2 medium, 2 large scale patches
  // train ferns to maximum depth of 5 levels 
  // subsample with 70% of the training samples used for each fern
  gpc::training::ForestSettings forestsettings(gpc::training::FernFactory(2,2,2,5) , 0.7);

  gpc::datasource::SintelOpticalFlow sintelOpticalFlow;

  // The container for the triplet samples, we'll laod from file
  // (faster than reextracting from original dataset each time we train ) 
  std::vector<gpc::training::Feature::GPCPatchTriplet> trainingData;

  cout << "Loading dataset" << std::endl;
  trainingData = sintelOpticalFlow.loadTrainingData(datasetPath);

  gpc::training::Forest gpcforest;
  gpcforest.trainAndExport(trainingData, forestsettings, zerooptimizer, forestPath);
}
