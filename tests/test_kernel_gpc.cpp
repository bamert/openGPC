#include <gtest/gtest.h>
#include <vector>
#include <random>
#include "gpc/forest.hpp"
#include "gpc/kernels/gpc.hpp"     // Naive version
#include "gpc/kernels/sobel.hpp" // Highway version
#include "gpc/kernels/gpc_hwy.hpp" // Highway version
#include "gpc/kernels/utils.hpp" // Highway version

TEST(Approval, GPCKernel) {
    auto file = std::filesystem::absolute(__FILE__);
    auto dir  = file.parent_path();
    std::filesystem::path forestPath = dir / ".." / "forests" / "defaultZeroForest.txt";

    const int width = 640;
    const int height = 480;
    const int radius = 2; // Typical for 5x5 bo
    const int threshold = 0; // Example threshold for binarization

    typedef gpc::inference::Forest GPCForest_t;
    GPCForest_t forest;
    gpc::inference::FilterMask fm =
        forest.readForest(forestPath, width, height);

    // 1. Prepare randomized input
    std::vector<uint8_t> input(width * height);
    std::mt19937 gen(42); 
    std::uniform_int_distribution<> dis(0, 255);
    for (auto& val : input) val = dis(gen);

    // 2. Prepare output buffers
    std::vector<uint8_t> grad(width * height, 0);
    std::vector<uint32_t> outNaive(width * height, 0);
    std::vector<uint32_t> outHighway(width * height, 0);

    // 3. Prepare gradient and fastmask
    ndb::sobelNaive(input.data(), grad.data(), width, height, threshold);

    // More prep
    std::vector<int> idx(grad.size());
    auto ff = [&](std::vector<int>& in, std::vector<int>& out, int m) {
        for (int i = 0; i < m; i++) {
            int x = in.data()[i] % width;
            int y = in.data()[i] / width;
            if (y >= 13 && y < height - 13 && x >= 13 && x < width - 13)
                out.push_back(in.data()[i]);
        }
    };
    int m;
    // mask indexing gradient pixels
    std::vector<int> fastmask;
    ndb::arr2ind(grad.data(), width * height, idx.data(), &m);
    ff(idx, fastmask, m);

    std::vector<int> tau;
    // 4. Run Naive version
    ndb::gpcFilterNaive(input.data(), grad.data(), outNaive.data(),
            fm.mask, fastmask, width, height);
    /*
     * fastmask.mask, fastmask.tau, idx
        fastmask.mask is.. imo the extraction pattern. lets se...
        it's filtermask! lol
        where idx is... preprocessed.mask. WTF..what is that lol
        */

    // 5. Run Highway version 
    //
    //ndb::gpcFilterSSE(input.data(), grad.data(), outHighway.data(), fastmask, tau, width, height);
    ndb::testing::gpc_hwy(input.data(), grad.data(), outHighway.data(),
            fastmask, tau, width, height);

    // 6. Compare results
    // We skip the border (radius) because different implementations 
    // might handle edges differently.
    for (int y = radius; y < height - radius; ++y) {
        for (int x = radius; x < width - radius; ++x) {
            int idx = y * width + x;
            ASSERT_EQ(outNaive[idx], outHighway[idx]) 
                << "Mismatch at (" << x << "," << y << ")";
        }
    }
}
