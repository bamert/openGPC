#include <hwy/highway.h>

#include <iostream>

#include "gpc/forest.hpp"
using namespace std;
std::vector<ndb::Descriptor> gpcFilterDense(
    uint8_t* in, const std::vector<int32_t>& fastmask, int width, int height) {
    uint32_t tmp;
    uint32_t usableW = width - 26;
    uint32_t usableH = height - 26;
    std::vector<ndb::Descriptor> out(usableW * usableH);
    int j = 0;
    for (int y = 13; y < height - 13; y++) {
        for (int x = 13; x < width - 13; x++) {
            tmp = 0;
            int idx = y * width + x;
            for (size_t i = 0; i < fastmask.size(); i += 2) {
                tmp <<= 1;  // shift by one
                if (*(in + idx + fastmask[i]) > *(in + idx + fastmask[i + 1]))
                    tmp++;  // set this test's result to 1
            }
            out[j] = ndb::Descriptor(ndb::Point(x, y), tmp);
            j++;
        }
    }
    return out;
}
int main(int argc, char** argv) {
    std::string forestPath = "../forests/defaultZeroForest.txt";
    std::string leftImgPath = "../data/middlebury/im0.png";
    std::string rightImgPath = "../data/middlebury/im1.png";

    if (argc == 4) {
        forestPath = argv[1];
        leftImgPath = argv[2];
        rightImgPath = argv[3];
    } else {
        cout << "Usage: " << argv[0]
             << " <forest path> <left image path> <right image path>" << endl;
        cout << "Trying defaults:" << endl;
        cout << "Forest path: " << forestPath << endl;
        cout << "Left image : " << leftImgPath << endl;
        cout << "Right image: " << rightImgPath << endl;
    }
    ndb::Buffer<uint8_t> simg, timg;

    typedef gpc::inference::Forest GPCForest_t;
    GPCForest_t forest;

#ifdef _INTRINSICS_SSE
    cout << "Using SSE intrinsics" << endl;
#endif

    gpc::inference::InferenceSettings inferencesettings =
        gpc::inference::InferenceSettings()
            .builder()
            .gradientThreshold(
                1)  // gradientthres 20: matching ~3ms, 2: matching: ~30ms.
            .verticalTolerance(
                0)               // 0px tolerance for rectified epipolar matches
            .dispHigh(128)       // limit disparities to 128
            .epipolarMode(true)  // match GPC states in epipolar mode. more
                                 // matches, lower accuracy than global
            .useHashtable(false);  // use sort method for matching. faster for
                                   // <100K descriptors

    // Load images
    simg.readPNG(leftImgPath);
    timg.readPNG(rightImgPath);

    // Get learned filter for the given image dimensions.
    gpc::inference::FilterMask fm =
        forest.readForest(forestPath, simg.cols(), simg.rows());

    gpc::inference::time_point t0 = gpc::inference::sysTick();

    gpc::inference::PreprocessedImage simgP =
        forest.preprocessImage(simg, inferencesettings);
    gpc::inference::PreprocessedImage timgP =
        forest.preprocessImage(timg, inferencesettings);
    gpc::inference::time_point t1 = gpc::inference::sysTick();

    // Match rectified stereo images
    std::vector<ndb::Support> supp =
        forest.rectifiedMatch(simgP, timgP, fm, inferencesettings);
    gpc::inference::time_point t2 = gpc::inference::sysTick();
    std::cout << "Number of features(s,t): " << simgP.mask.size() << ","
              << timgP.mask.size() << std::endl;
    std::cout << "Number of matches: " << supp.size() << std::endl;
    std::cout << "Preprocessing time: " << gpc::inference::tickToMs(t1, t0)
              << " ms" << std::endl;
    std::cout << "Matching time: " << gpc::inference::tickToMs(t2, t1) << " ms"
              << std::endl;
    /*
    std::vector<ndb::Descriptor> statesSrc = forest.evalFastMaskOnSubsetSSE(
        simgP.smooth, simgP.grad, simgP.mask, fm, inferencesettings);
    std::vector<ndb::Descriptor> statesTar = forest.evalFastMaskOnSubsetSSE(
        timgP.smooth, timgP.grad, timgP.mask, fm, inferencesettings);
    */

    std::vector<ndb::Descriptor> statesSrc = gpcFilterDense(
        simgP.smooth.data(), fm.mask, simgP.smooth.cols(), simgP.smooth.rows());
    std::vector<ndb::Descriptor> statesTar = gpcFilterDense(
        timgP.smooth.data(), fm.mask, timgP.smooth.cols(), timgP.smooth.rows());

    ndb::Descriptor::serialize("statesSrcLargeS.txt", statesSrc);
    ndb::Descriptor::serialize("statesTarLargeS.txt", statesTar);
}
