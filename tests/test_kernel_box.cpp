#include <gtest/gtest.h>
#include <vector>
#include <random>
#include "gpc/kernels/box.hpp"     // Naive version
#include "gpc/kernels/box_hwy.hpp" // Highway version

TEST(Approval, BoxKernel) {
    const int width = 640;
    const int height = 480;
    const int radius = 2; // Typical for 5x5 box

    // 1. Prepare randomized input
    std::vector<uint8_t> input(width * height);
    std::mt19937 gen(42); 
    std::uniform_int_distribution<> dis(0, 255);
    for (auto& val : input) val = dis(gen);

    // 2. Prepare output buffers
    std::vector<uint8_t> outNaive(width * height, 0);
    std::vector<uint8_t> outHighway(width * height, 0);

    // 3. Run Naive version
    ndb::boxNaive(input.data(), outNaive.data(), width, height);

    // 4. Run Highway version (only if compiled for the target)
#if defined(HWY_TARGET) && HWY_TARGET == HWY_NEON
    ndb::BoxFilter(input.data(), outHighway.data(), width, height);
#else
    ndb::boxNaive(input.data(), outNaive.data(), width, height);
    // Fallback if the specific NEON namespace isn't exposed
    //ndb::testing::box_hwy(input.data(), outHighway.data(), width, height);

#endif

    // 5. Compare results
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
