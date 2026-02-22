#define HWY_TARGET HWY_NEON 
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE(); 
namespace ndb {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;


void BoxKernel(const uint8_t* HWY_RESTRICT in, uint8_t* HWY_RESTRICT blurred, int width, int height) {
    const hn::ScalableTag<uint8_t> d8;
    const hn::Half<decltype(d8)> d8_h;
    const hn::Rebind<uint16_t, decltype(d8_h)> d16;
    
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
            auto v00 = hn::LoadU(d8, r0+x); auto v01 = hn::LoadU(d8, r0+x+1); auto v02 = hn::LoadU(d8, r0+x+2);
            auto v10 = hn::LoadU(d8, r1+x); auto v11 = hn::LoadU(d8, r1+x+1); auto v12 = hn::LoadU(d8, r1+x+2);
            auto v20 = hn::LoadU(d8, r2+x); auto v21 = hn::LoadU(d8, r2+x+1); auto v22 = hn::LoadU(d8, r2+x+2);
            auto v30 = hn::LoadU(d8, r3+x); auto v31 = hn::LoadU(d8, r3+x+1); auto v32 = hn::LoadU(d8, r3+x+2);

            // Lower Half Math
            auto s1_lo = hn::Add(hn::PromoteTo(d16, hn::LowerHalf(v11)), hn::Add(hn::PromoteTo(d16, hn::LowerHalf(v10)), hn::PromoteTo(d16, hn::LowerHalf(v12))));
            auto s2_lo = hn::Add(hn::PromoteTo(d16, hn::LowerHalf(v21)), hn::Add(hn::PromoteTo(d16, hn::LowerHalf(v20)), hn::PromoteTo(d16, hn::LowerHalf(v22))));
            
            auto row0_lo = hn::Add(hn::Add(hn::PromoteTo(d16, hn::LowerHalf(v01)), hn::Add(hn::PromoteTo(d16, hn::LowerHalf(v00)), hn::PromoteTo(d16, hn::LowerHalf(v02)))), hn::Add(s1_lo, s2_lo));
            auto row1_lo = hn::Add(hn::Add(hn::PromoteTo(d16, hn::LowerHalf(v31)), hn::Add(hn::PromoteTo(d16, hn::LowerHalf(v30)), hn::PromoteTo(d16, hn::LowerHalf(v32)))), hn::Add(s1_lo, s2_lo));

            // Upper Half Math
            auto s1_hi = hn::Add(hn::PromoteTo(d16, hn::UpperHalf(d8_h, v11)), hn::Add(hn::PromoteTo(d16, hn::UpperHalf(d8_h, v10)), hn::PromoteTo(d16, hn::UpperHalf(d8_h, v12))));
            auto s2_hi = hn::Add(hn::PromoteTo(d16, hn::UpperHalf(d8_h, v21)), hn::Add(hn::PromoteTo(d16, hn::UpperHalf(d8_h, v20)), hn::PromoteTo(d16, hn::UpperHalf(d8_h, v22))));
            
            auto row0_hi = hn::Add(hn::Add(hn::PromoteTo(d16, hn::UpperHalf(d8_h, v01)), hn::Add(hn::PromoteTo(d16, hn::UpperHalf(d8_h, v00)), hn::PromoteTo(d16, hn::UpperHalf(d8_h, v02)))), hn::Add(s1_hi, s2_hi));
            auto row1_hi = hn::Add(hn::Add(hn::PromoteTo(d16, hn::UpperHalf(d8_h, v31)), hn::Add(hn::PromoteTo(d16, hn::UpperHalf(d8_h, v30)), hn::PromoteTo(d16, hn::UpperHalf(d8_h, v32)))), hn::Add(s1_hi, s2_hi));

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
//#if defined(HWY_TARGET) && HWY_TARGET == HWY_NEON
    void box_hwy(uint8_t* in, uint8_t* blurred, int width, int height) {
        //ndb::N_NEON::BoxKernel(in, blurred, width, height);
        HWY_STATIC_DISPATCH(BoxKernel)(in, blurred, width, height);
    }
//#endif
}
}
