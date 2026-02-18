#include <iostream>
#include <hwy/highway.h>

#include "gpc/forest.hpp"
using namespace std;
void test_hwy_neon() {
    namespace hn = hwy::HWY_NAMESPACE;
    
    // d is a "descriptor" for a vector of 8-bit unsigned ints
    const hn::ScalableTag<uint8_t> d;
    
    // If this is NEON, hn::Lanes(d) will be 16
    size_t lanes = hn::Lanes(d);
    
    auto v1 = hn::Set(d, 10);
    auto v2 = hn::Set(d, 20);
    auto res = hn::Add(v1, v2); // res lanes all contain 30
    
    std::cout << "--- Highway Status ---" << std::endl;
    std::cout << "Target: " << hwy::TargetName(hwy::SupportedTargets()) << std::endl;
    std::cout << "Vector lanes (uint8): " << lanes << std::endl;
    std::cout << "----------------------" << std::endl;
}
int main(int argc, char** argv) {
    std::string forestPath = "../../forests/defaultZeroForest.txt";
    std::string leftImgPath = "../../data/kitti/training/image_0/000000_10.png";
    std::string rightImgPath =
        "../../data/kitti/training/image_1/000000_10.png";

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
            .gradientThreshold(2) // gradientthres 20: matching ~3ms, 2: matching: ~30ms. 
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

    for(int i = 0; i<10000; i++) {
    // Preprocess images (box filter, sobel filter, indices of high gradient
    // pixels)

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
    std::cout << "Preprocessing time: " << gpc::inference::tickToMs(t1, t0) << " ms" << std::endl;
    std::cout << "Matching time: " << gpc::inference::tickToMs(t2, t1) << " ms" << std::endl;
    }
    test_hwy_neon();
}
