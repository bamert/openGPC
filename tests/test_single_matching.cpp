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
std::vector<ndb::Descriptor> getSrcDescriptors() {
    return ndb::Descriptor::deserialize("statesSrc.txt", true);
}

std::vector<ndb::Descriptor> getTarDescriptors() {
    return ndb::Descriptor::deserialize("statesTar.txt", false);
}


TEST(A,B) {
    std::vector<ndb::Descriptor> srcOriginal = getSrcDescriptors(); 
    std::vector<ndb::Descriptor> tarOriginal = getTarDescriptors();
    std::vector<ndb::Descriptor> srcBaseline = srcOriginal;
    std::vector<ndb::Descriptor> tarBaseline = tarOriginal;
    std::vector<ndb::Descriptor> srcAlt = srcOriginal;
    std::vector<ndb::Descriptor> tarAlt = tarOriginal;
    
    // Baseline
    // To write a test for this we'd actually need to get the ids of the sources back, not just the final matches.
    std::vector<ndb::Correspondence> 
        matches = gpc::inference::Forest::findCorrespondences(srcBaseline, tarBaseline);


    // Alternative method
    gpc::inference::SoAFramePersistentSingleSlab srcFrame, tarFrame;
    srcFrame.preallocate(srcOriginal.size()); // size known
    tarFrame.preallocate(tarOriginal.size());

    std::vector<uint32_t> resultSrc, resultTar;
    resultSrc.reserve(srcOriginal.size()/10);
    resultTar.reserve(tarOriginal.size()/10);
    gpc::inference::Forest::prepareSoAFramesPersistentSingleSlabUnordered(srcAlt, tarAlt, srcFrame, tarFrame);
    gpc::inference::Forest::matchPipelinedBranchlessPreallocateSingleSlabUnordered(srcFrame, tarFrame, resultSrc, resultTar);

    // Ensure ID pairings of (resultSrc, resultTar) match the naive version. 
    // We ignore exact matching for now and just expect the count to be the same
    EXPECT_EQ(matches.size(), resultSrc.size());
    EXPECT_EQ(matches.size(), resultTar.size());
}
