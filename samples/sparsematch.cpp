#include <iostream>
#include "gpc/inference.hpp"
using namespace std;
int main(int argc, char** argv) {
    std::string forestPath = "../../forests/defaultZeroForest.txt";
    std::string leftImgPath = "../../data/kitti/img0l.png";
    std::string rightImgPath = "../../data/kitti/img0r.png";

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
        cout << "Right image: " << leftImgPath << endl;
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
            .gradientThreshold(5)
            .verticalTolerance(0)  // 0px tolerance for rectified epipolar matches
            .dispHigh(128)         // limit disparities to 128
            .epipolarMode(true)  // match GPC states in epipolar mode. more matches, lower
                                 // accuracy than global
            .useHashtable(
                false);  // use sort method for matching. faster for <100K descriptors

    // Load images
    if (simg.readPNG(leftImgPath) || timg.readPNG(rightImgPath)) {
        cout << "No image data \n";
        return -1;
    }
    // Get learned filter for the given image dimensions.
    GPCForest_t::FilterMask fm = forest.readForest(forestPath, simg.cols(), simg.rows());

    // Preprocess images (box filter, sobel filter, indices of high gradient pixels)
    gpc::inference::time_point t0 = gpc::inference::sysTick();
    GPCForest_t::PreprocessedImage simgP =
        forest.preprocessImage(simg, inferencesettings);
    GPCForest_t::PreprocessedImage timgP =
        forest.preprocessImage(timg, inferencesettings);
    gpc::inference::time_point t1 = gpc::inference::sysTick();

    // Match rectified stereo images
    std::vector<ndb::Support> supp =
        forest.rectifiedMatch(simgP, timgP, fm, inferencesettings);
    gpc::inference::time_point t2 = gpc::inference::sysTick();
    cout << "tPreprocess: " << gpc::inference::tickToMs(t1, t0) << " ms"
         << ", tMatch: " << gpc::inference::tickToMs(t2, t1) << " ms"
         << ", num matches:" << supp.size() << endl;

    // Output sparse disparities overlayed on left input image
    ndb::Buffer<ndb::RGBColor> renderDisp;
    renderDisp = ndb::getDisparityVisualization(simg, supp);
    renderDisp.writePNGRGB("disparity.png");
}
