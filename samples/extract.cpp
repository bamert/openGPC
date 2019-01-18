#include <iostream>
#include "gpc/training.hpp"
using namespace std;

int main(int argc, char** argv) {
 //Parse arguments
  std::string sintelPath = "../../data/MPI-Sintel-complete"; 
  std::string outputFile  = "../../data/SintelOpticalFlow-extracted.bin";

  if(argc == 3){
    sintelPath = argv[1];
    outputFile = argv[2];
  }else{
    cout << "Usage: " << argv[0] << " <sintel training set root dir path> <extracted dataset path>" << endl;
    cout << "Trying defaults:" << endl;
    cout << "Sintel dataset location    : " << sintelPath << endl;
    cout << "Export extracted dataset to: " << outputFile << endl;
  }

  typedef gpc::training::Feature  GPCFeature_t;
  // The training set can be either SintelOpticalFlow or SintelStereo
  gpc::datasource::SintelOpticalFlow sintelOpticalFlow(sintelPath);

  // The container for the triplet samples we will extract
  std::vector<typename GPCFeature_t::GPCPatchTriplet> trainingData;

  cout << "Extracting samples" << std::endl;
  
  // Extract up to 1000 triplets per image. 
  // Sample negative patch from annulus with radii 20,40 centered at positive match
  trainingData = sintelOpticalFlow.extractTrainingData(1000,20,40);
  // Store training set to file
  sintelOpticalFlow.storeTrainingData(trainingData,outputFile);
}
