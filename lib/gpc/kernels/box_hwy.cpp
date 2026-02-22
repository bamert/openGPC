#define HWY_TARGET HWY_NEON 
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE(); 
namespace ndb {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;


void BoxKernel(const uint8_t* HWY_RESTRICT in, uint8_t* HWY_RESTRICT blurred, int width, int height) {
    const hn::ScalableTag<uint8_t> d8;
    // We need d16 to be the "Promoted" version of the half-width d8 to stay lane-consistent
    const hn::Rebind<uint16_t, hn::Half<decltype(d8)>> d16;
    
    const size_t N = hn::Lanes(d8);
    const auto divisor = hn::Set(d16, (uint16_t)7282);

    for (int y = 1; y < height - 2; y += 2) {
        const uint8_t* r0 = in + (y - 1) * width;
        const uint8_t* r1 = in + y * width;
        const uint8_t* r2 = in + (y + 1) * width;
        const uint8_t* r3 = in + (y + 2) * width;
        
        uint8_t* out0 = blurred + y * width + 1;
        uint8_t* out1 = blurred + (y + 1) * width + 1;

        for (int x = 0; x < width; x += N) {
            auto v0_0 = hn::LoadU(d8, r0 + x); auto v0_1 = hn::LoadU(d8, r0 + x + 1); auto v0_2 = hn::LoadU(d8, r0 + x + 2);
            auto v1_0 = hn::LoadU(d8, r1 + x); auto v1_1 = hn::LoadU(d8, r1 + x + 1); auto v1_2 = hn::LoadU(d8, r1 + x + 2);
            auto v2_0 = hn::LoadU(d8, r2 + x); auto v2_1 = hn::LoadU(d8, r2 + x + 1); auto v2_2 = hn::LoadU(d8, r2 + x + 2);
            auto v3_0 = hn::LoadU(d8, r3 + x); auto v3_1 = hn::LoadU(d8, r3 + x + 1); auto v3_2 = hn::LoadU(d8, r3 + x + 2);

            // Helper to sum 3 promoted pixels
            auto sum3 = [&](auto v0, auto v1, auto v2) {
                return hn::Add(v1, hn::Add(v0, v2));
            };

            // LOWER HALF
            auto s1_lo = sum3(hn::PromoteTo(d16, hn::LowerHalf(v1_0)), hn::PromoteTo(d16, hn::LowerHalf(v1_1)), hn::PromoteTo(d16, hn::LowerHalf(v1_2)));
            auto s2_lo = sum3(hn::PromoteTo(d16, hn::LowerHalf(v2_0)), hn::PromoteTo(d16, hn::LowerHalf(v2_1)), hn::PromoteTo(d16, hn::LowerHalf(v2_2)));
            
            auto row0_lo = hn::Add(sum3(hn::PromoteTo(d16, hn::LowerHalf(v0_0)), hn::PromoteTo(d16, hn::LowerHalf(v0_1)), hn::PromoteTo(d16, hn::LowerHalf(v0_2))), hn::Add(s1_lo, s2_lo));
            auto row1_lo = hn::Add(sum3(hn::PromoteTo(d16, hn::LowerHalf(v3_0)), hn::PromoteTo(d16, hn::LowerHalf(v3_1)), hn::PromoteTo(d16, hn::LowerHalf(v3_2))), hn::Add(s1_lo, s2_lo));

            // UPPER HALF
            auto s1_hi = sum3(hn::PromoteTo(d16, hn::UpperHalf(d8, v1_0)), hn::PromoteTo(d16, hn::UpperHalf(d8, v1_1)), hn::PromoteTo(d16, hn::UpperHalf(d8, v1_2)));
            auto s2_hi = sum3(hn::PromoteTo(d16, hn::UpperHalf(d8, v2_0)), hn::PromoteTo(d16, hn::UpperHalf(d8, v2_1)), hn::PromoteTo(d16, hn::UpperHalf(d8, v2_2)));
            
            auto row0_hi = hn::Add(sum3(hn::PromoteTo(d16, hn::UpperHalf(d8, v0_0)), hn::PromoteTo(d16, hn::UpperHalf(d8, v0_1)), hn::PromoteTo(d16, hn::UpperHalf(d8, v0_2))), hn::Add(s1_hi, s2_hi));
            auto row1_hi = hn::Add(sum3(hn::PromoteTo(d16, hn::UpperHalf(d8, v3_0)), hn::PromoteTo(d16, hn::UpperHalf(d8, v3_1)), hn::PromoteTo(d16, hn::UpperHalf(d8, v3_2))), hn::Add(s1_hi, s2_hi));

            // Perform normalization and store using OrderedDemote2To
            hn::StoreU(hn::OrderedDemote2To(d8, hn::MulHigh(row0_lo, divisor), hn::MulHigh(row0_hi, divisor)), d8, out0 + x);
            hn::StoreU(hn::OrderedDemote2To(d8, hn::MulHigh(row1_lo, divisor), hn::MulHigh(row1_hi, divisor)), d8, out1 + x);
        }
    }
}
} // namespace HWY_NAMESPACE
} // namespace ndb
HWY_AFTER_NAMESPACE();

namespace ndb {
namespace testing {
#if defined(HWY_TARGET) && HWY_TARGET == HWY_NEON
    void box_hwy(uint8_t* in, uint8_t* blurred, int width, int height) {
        ndb::N_NEON::BoxKernel(in, blurred, width, height);
    }
#endif
}
}
