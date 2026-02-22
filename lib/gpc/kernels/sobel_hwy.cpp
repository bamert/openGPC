#define HWY_TARGET HWY_NEON 
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE(); 
namespace ndb {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;


void SobelKernel(const uint8_t* HWY_RESTRICT in, uint8_t* HWY_RESTRICT gradient, 
                 int width, int height, uint8_t threshold) {
    const hn::ScalableTag<uint8_t> d8;
    const hn::Half<decltype(d8)> d8_h; 
    const hn::Rebind<int16_t, decltype(d8_h)> d16; 

    const size_t N = hn::Lanes(d8);
    const auto divisor = hn::Set(d16, (int16_t)7282); 
    const auto threshSq = hn::Set(d16, (int16_t)(threshold * threshold));
    const auto v255 = hn::Set(d16, 255);
    const auto v0 = hn::Zero(d16);

    for (int y = 1; y < height - 1; ++y) {
        const uint8_t* r0 = in + (y - 1) * width;
        const uint8_t* r1 = in + y * width;
        const uint8_t* r2 = in + (y + 1) * width;
        uint8_t* out = gradient + y * width + 1;

        for (int x = 0; x < width; x += N) {
            auto v11 = hn::LoadU(d8, r0 + x); auto v12 = hn::LoadU(d8, r0 + x + 1); auto v13 = hn::LoadU(d8, r0 + x + 2);
            auto v21 = hn::LoadU(d8, r1 + x);                                      auto v23 = hn::LoadU(d8, r1 + x + 2);
            auto v31 = hn::LoadU(d8, r2 + x); auto v32 = hn::LoadU(d8, r2 + x + 1); auto v33 = hn::LoadU(d8, r2 + x + 2);

            auto process = [&](auto p11, auto p12, auto p13, auto p21, auto p23, auto p31, auto p32, auto p33) {
                auto sx = hn::Sub(hn::Add(hn::Add(p11, p31), hn::Add(p21, p21)), 
                                  hn::Add(hn::Add(p13, p33), hn::Add(p23, p23)));
                sx = hn::MulHigh(sx, divisor);
                auto sy = hn::Sub(hn::Add(hn::Add(p11, p13), hn::Add(p12, p12)), 
                                  hn::Add(hn::Add(p31, p33), hn::Add(p32, p32)));
                sy = hn::MulHigh(sy, divisor);
                auto mag = hn::Add(hn::Mul(sx, sx), hn::Mul(sy, sy));
                return hn::IfThenElse(hn::Gt(mag, threshSq), v255, v0);
            };

            // Process Lower Half
            auto res_lo = process(
                hn::PromoteTo(d16, hn::LowerHalf(v11)), hn::PromoteTo(d16, hn::LowerHalf(v12)), hn::PromoteTo(d16, hn::LowerHalf(v13)),
                hn::PromoteTo(d16, hn::LowerHalf(v21)), hn::PromoteTo(d16, hn::LowerHalf(v23)),
                hn::PromoteTo(d16, hn::LowerHalf(v31)), hn::PromoteTo(d16, hn::LowerHalf(v32)), hn::PromoteTo(d16, hn::LowerHalf(v33)));

            // Process Upper Half - Using correct d8_h tag
            auto res_hi = process(
                hn::PromoteTo(d16, hn::UpperHalf(d8_h, v11)), hn::PromoteTo(d16, hn::UpperHalf(d8_h, v12)), hn::PromoteTo(d16, hn::UpperHalf(d8_h, v13)),
                hn::PromoteTo(d16, hn::UpperHalf(d8_h, v21)), hn::PromoteTo(d16, hn::UpperHalf(d8_h, v23)),
                hn::PromoteTo(d16, hn::UpperHalf(d8_h, v31)), hn::PromoteTo(d16, hn::UpperHalf(d8_h, v32)), hn::PromoteTo(d16, hn::UpperHalf(d8_h, v33)));

            hn::StoreU(hn::OrderedDemote2To(d8, res_lo, res_hi), d8, out + x);
        }
    }
}
} // namespace HWY_NAMESPACE
} // namespace ndb
HWY_AFTER_NAMESPACE();

namespace ndb {
namespace testing {
//#if defined(HWY_TARGET) && HWY_TARGET == HWY_NEON
    void sobel_hwy(uint8_t* in, uint8_t* blurred, int width, int height, uint8_t threshold) {
        //ndb::N_NEON::SobelKernel(in, blurred, width, height, threshold);
        HWY_STATIC_DISPATCH(SobelKernel)(in, blurred, width, height, threshold);
    }
//#endif  
}
}
