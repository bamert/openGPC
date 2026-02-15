
// We define the target BEFORE including highway.h
// On Mac, this forces Highway to use NEON mode without the inclusion loop.
#define HWY_TARGET HWY_NEON 
#include <hwy/highway.h>

// We skip foreach_target.h entirely to avoid the "redefinition" and "path" errors.

HWY_BEFORE_NAMESPACE(); 
namespace ndb {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void BoxKernelNaive(const uint8_t* HWY_RESTRICT in, uint8_t* HWY_RESTRICT blurred, int width, int height) {
    const hn::ScalableTag<uint8_t> d8;
    const hn::ScalableTag<uint16_t> d16;
    const size_t N = hn::Lanes(d8);
    const auto divisor = hn::Set(d16, (uint16_t)7282); // 65536 / 9

    for (int y = 1; y < height - 1; ++y) {
        const uint8_t* r0 = in + (y - 1) * width;
        const uint8_t* r1 = in + y * width;
        const uint8_t* r2 = in + (y + 1) * width;
        
        uint8_t* out_row = blurred + y * width + 1;

        for (int x = 0; x < width; x += N) {
            // Row 0
            auto v11 = hn::LoadU(d8, r0 + x);
            auto v12 = hn::LoadU(d8, r0 + x + 1);
            auto v13 = hn::LoadU(d8, r0 + x + 2);
            
            // Row 1
            auto v21 = hn::LoadU(d8, r1 + x);
            auto v22 = hn::LoadU(d8, r1 + x + 1);
            auto v23 = hn::LoadU(d8, r1 + x + 2);

            // Row 2
            auto v31 = hn::LoadU(d8, r2 + x);
            auto v32 = hn::LoadU(d8, r2 + x + 1);
            auto v33 = hn::LoadU(d8, r2 + x + 2);

            // Vertical sums first (3 instructions per half-vector)
            auto sum_col1_lo = hn::Add(hn::PromoteLowerTo(d16, v11), hn::Add(hn::PromoteLowerTo(d16, v21), hn::PromoteLowerTo(d16, v31)));
            auto sum_col1_hi = hn::Add(hn::PromoteUpperTo(d16, v11), hn::Add(hn::PromoteUpperTo(d16, v21), hn::PromoteUpperTo(d16, v31)));

            auto sum_col2_lo = hn::Add(hn::PromoteLowerTo(d16, v12), hn::Add(hn::PromoteLowerTo(d16, v22), hn::PromoteLowerTo(d16, v32)));
            auto sum_col2_hi = hn::Add(hn::PromoteUpperTo(d16, v12), hn::Add(hn::PromoteUpperTo(d16, v22), hn::PromoteUpperTo(d16, v32)));

            auto sum_col3_lo = hn::Add(hn::PromoteLowerTo(d16, v13), hn::Add(hn::PromoteLowerTo(d16, v23), hn::PromoteLowerTo(d16, v33)));
            auto sum_col3_hi = hn::Add(hn::PromoteUpperTo(d16, v13), hn::Add(hn::PromoteUpperTo(d16, v23), hn::PromoteUpperTo(d16, v33)));

            // Horizontal accumulation
            auto total_lo = hn::Add(sum_col1_lo, hn::Add(sum_col2_lo, sum_col3_lo));
            auto total_hi = hn::Add(sum_col1_hi, hn::Add(sum_col2_hi, sum_col3_hi));

            // Fixed-point division by 9
            auto res_lo = hn::MulHigh(total_lo, divisor);
            auto res_hi = hn::MulHigh(total_hi, divisor);
            
            hn::StoreU(hn::Combine(d8, hn::DemoteTo(d8, res_hi), hn::DemoteTo(d8, res_lo)), d8, out_row + x);
        }
    }
}
void BoxKernel(const uint8_t* HWY_RESTRICT in, uint8_t* HWY_RESTRICT blurred, int width, int height) {
    const hn::ScalableTag<uint8_t> d8;
    const hn::ScalableTag<uint16_t> d16;
    const size_t N = hn::Lanes(d8);
    const auto divisor = hn::Set(d16, (uint16_t)7282);

    // We process two output rows at a time (y and y+1)
    // This requires 4 input rows (r0, r1, r2, r3)
    for (int y = 1; y < height - 2; y += 2) {
        const uint8_t* r0 = in + (y - 1) * width;
        const uint8_t* r1 = in + y * width;
        const uint8_t* r2 = in + (y + 1) * width;
        const uint8_t* r3 = in + (y + 2) * width;
        
        uint8_t* out0 = blurred + y * width + 1;
        uint8_t* out1 = blurred + (y + 1) * width + 1;

        for (int x = 0; x < width; x += N) {
            // Load all 4 rows needed for 2 output rows
            auto v0_0 = hn::LoadU(d8, r0 + x);
            auto v0_1 = hn::LoadU(d8, r0 + x + 1);
            auto v0_2 = hn::LoadU(d8, r0 + x + 2);

            auto v1_0 = hn::LoadU(d8, r1 + x);
            auto v1_1 = hn::LoadU(d8, r1 + x + 1);
            auto v1_2 = hn::LoadU(d8, r1 + x + 2);

            auto v2_0 = hn::LoadU(d8, r2 + x);
            auto v2_1 = hn::LoadU(d8, r2 + x + 1);
            auto v2_2 = hn::LoadU(d8, r2 + x + 2);

            auto v3_0 = hn::LoadU(d8, r3 + x);
            auto v3_1 = hn::LoadU(d8, r3 + x + 1);
            auto v3_2 = hn::LoadU(d8, r3 + x + 2);

            // Vertical sums for Row Pair 1 (Rows 0, 1, 2)
            // Vertical sums for Row Pair 2 (Rows 1, 2, 3)
            // Note: Rows 1 and 2 are REUSED.
            
            auto s1_lo = hn::Add(hn::PromoteLowerTo(d16, v1_1), hn::Add(hn::PromoteLowerTo(d16, v1_0), hn::PromoteLowerTo(d16, v1_2)));
            auto s2_lo = hn::Add(hn::PromoteLowerTo(d16, v2_1), hn::Add(hn::PromoteLowerTo(d16, v2_0), hn::PromoteLowerTo(d16, v2_2)));
            
            // Output Row 0 logic
            auto s0_lo = hn::Add(hn::PromoteLowerTo(d16, v0_1), hn::Add(hn::PromoteLowerTo(d16, v0_0), hn::PromoteLowerTo(d16, v0_2)));
            auto row0_lo = hn::Add(s0_lo, hn::Add(s1_lo, s2_lo));

            // Output Row 1 logic
            auto s3_lo = hn::Add(hn::PromoteLowerTo(d16, v3_1), hn::Add(hn::PromoteLowerTo(d16, v3_0), hn::PromoteLowerTo(d16, v3_2)));
            auto row1_lo = hn::Add(s3_lo, hn::Add(s1_lo, s2_lo));

            // Repeat for high bits...
            auto s1_hi = hn::Add(hn::PromoteUpperTo(d16, v1_1), hn::Add(hn::PromoteUpperTo(d16, v1_0), hn::PromoteUpperTo(d16, v1_2)));
            auto s2_hi = hn::Add(hn::PromoteUpperTo(d16, v2_1), hn::Add(hn::PromoteUpperTo(d16, v2_0), hn::PromoteUpperTo(d16, v2_2)));
            
            auto s0_hi = hn::Add(hn::PromoteUpperTo(d16, v0_1), hn::Add(hn::PromoteUpperTo(d16, v0_0), hn::PromoteUpperTo(d16, v0_2)));
            auto row0_hi = hn::Add(s0_hi, hn::Add(s1_hi, s2_hi));

            auto s3_hi = hn::Add(hn::PromoteUpperTo(d16, v3_1), hn::Add(hn::PromoteUpperTo(d16, v3_0), hn::PromoteUpperTo(d16, v3_2)));
            auto row1_hi = hn::Add(s3_hi, hn::Add(s1_hi, s2_hi));

            // Store both rows
            hn::StoreU(hn::Combine(d8, hn::DemoteTo(d8, hn::MulHigh(row0_hi, divisor)), 
                                       hn::DemoteTo(d8, hn::MulHigh(row0_lo, divisor))), d8, out0 + x);
            hn::StoreU(hn::Combine(d8, hn::DemoteTo(d8, hn::MulHigh(row1_hi, divisor)), 
                                       hn::DemoteTo(d8, hn::MulHigh(row1_lo, divisor))), d8, out1 + x);
        }
    }
}

} // namespace HWY_NAMESPACE
} // namespace ndb
HWY_AFTER_NAMESPACE();

namespace ndb {
namespace testing {
    void box_hwy(uint8_t* in, uint8_t* blurred, int width, int height) {
        // We call ghwthe NEON version directly. 
        // Highway maps HWY_NAMESPACE to N_NEON because of our #define above.
        ndb::N_NEON::BoxKernel(in, blurred, width, height);
    }
}
}
