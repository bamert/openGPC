#define APPROVALS_GOOGLETEST
#include <ApprovalTests.hpp>     
#include <gtest/gtest.h>
#include "gpc/forest.hpp"


TEST(Approval, Inference)
{
    auto file = std::filesystem::absolute(__FILE__);
    auto dir  = file.parent_path();
    std::filesystem::path forestPath = dir / ".." / "forests" / "defaultZeroForest.txt";
    
    std::string leftImgPath = (dir / ".." / "data" / "middlebury" / "im0.png").string();
    std::string rightImgPath = (dir / ".." / "data" / "middlebury" / "im1.png").string();

    ndb::Buffer<uint8_t> simg, timg;

    typedef gpc::inference::Forest GPCForest_t;
    GPCForest_t forest;

    gpc::inference::InferenceSettings inferencesettings =
        gpc::inference::InferenceSettings()
            .builder()
            .gradientThreshold(5)
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

    gpc::inference::PreprocessedImage simgP =
        forest.preprocessImage(simg, inferencesettings);
    gpc::inference::PreprocessedImage timgP =
        forest.preprocessImage(timg, inferencesettings);

    // Match rectified stereo images
    std::vector<ndb::Support> supp =
        forest.rectifiedMatch(simgP, timgP, fm, inferencesettings);
    std::sort(supp.begin(), supp.end());

    std::stringstream ss;
    ss << supp;
    EXPECT_EQ(866, supp.size());
    ApprovalTests::Approvals::verify(ss.str());
}
