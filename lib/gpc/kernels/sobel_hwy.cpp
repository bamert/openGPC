//#define HWY_TARGET HWY_NEON 
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE(); 
namespace ndb {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;
void SobelKernelNoDiv(const uint8_t* HWY_RESTRICT in, uint8_t* HWY_RESTRICT gradient, 
                 int width, int height, uint8_t threshold) {
    const hn::ScalableTag<uint8_t> d8;
    const hn::Half<decltype(d8)> d8_h; 
    const hn::Rebind<int16_t, decltype(d8_h)> d16;
    // d32 has half the lanes of d16
    const hn::Rebind<int32_t, hn::Half<decltype(d16)>> d32;

    const size_t N = hn::Lanes(d8);
    const auto vDivMult = hn::Set(d16, (int16_t)7282); 
    const auto vThreshSq = hn::Set(d32, (int32_t)threshold * threshold);
    const auto v255_16 = hn::Set(d16, (int16_t)255);
    const auto v255_8 = hn::Set(d8, (uint8_t)255);
    const auto v0_8 = hn::Zero(d8);

    for (int y = 1; y < height - 1; ++y) {
        const uint8_t* r0 = in + (y - 1) * width;
        const uint8_t* r1 = in + y * width;
        const uint8_t* r2 = in + (y + 1) * width;
        uint8_t* out = gradient + y * width + 1;

        for (int x = 0; x < width; x += N) {
            auto v11 = hn::LoadU(d8, r0 + x); auto v12 = hn::LoadU(d8, r0 + x + 1); auto v13 = hn::LoadU(d8, r0 + x + 2);
            auto v21 = hn::LoadU(d8, r1 + x);                                       auto v23 = hn::LoadU(d8, r1 + x + 2);
            auto v31 = hn::LoadU(d8, r2 + x); auto v32 = hn::LoadU(d8, r2 + x + 1); auto v33 = hn::LoadU(d8, r2 + x + 2);

            // Helper to process 8 pixels into a 16-bit mask-like result
            auto process_half = [&](auto p11, auto p12, auto p13, auto p21, auto p23, auto p31, auto p32, auto p33) {
                // Sobel derivatives in 16-bit
                auto sx16 = hn::MulHigh(hn::Sub(hn::Add(hn::Add(p11, p31), hn::Add(p21, p21)), 
                                                hn::Add(hn::Add(p13, p33), hn::Add(p23, p23))), vDivMult);
                auto sy16 = hn::MulHigh(hn::Sub(hn::Add(hn::Add(p11, p13), hn::Add(p12, p12)), 
                                                hn::Add(hn::Add(p31, p33), hn::Add(p32, p32))), vDivMult);

                // Magnitude squared in 32-bit
                auto sx_lo = hn::PromoteLowerTo(d32, sx16);
                auto sy_lo = hn::PromoteLowerTo(d32, sy16);
                auto mag_lo = hn::Add(hn::Mul(sx_lo, sx_lo), hn::Mul(sy_lo, sy_lo));

                auto sx_hi = hn::PromoteUpperTo(d32, sx16);
                auto sy_hi = hn::PromoteUpperTo(d32, sy16);
                auto mag_hi = hn::Add(hn::Mul(sx_hi, sx_hi), hn::Mul(sy_hi, sy_hi));

                // Comparison in 32-bit, returning 16-bit values (0 or 255) to avoid mask issues
                auto m_lo = hn::IfThenElse(hn::Gt(mag_lo, vThreshSq), hn::Set(d32, 255), hn::Zero(d32));
                auto m_hi = hn::IfThenElse(hn::Gt(mag_hi, vThreshSq), hn::Set(d32, 255), hn::Zero(d32));

                return hn::OrderedDemote2To(d16, m_lo, m_hi);
            };

            // Process halves using standard Highway promotion
            auto res_lo = process_half(
                hn::PromoteLowerTo(d16, v11), hn::PromoteLowerTo(d16, v12), hn::PromoteLowerTo(d16, v13),
                hn::PromoteLowerTo(d16, v21), hn::PromoteLowerTo(d16, v23),
                hn::PromoteLowerTo(d16, v31), hn::PromoteLowerTo(d16, v32), hn::PromoteLowerTo(d16, v33));

            auto res_hi = process_half(
                hn::PromoteUpperTo(d16, v11), hn::PromoteUpperTo(d16, v12), hn::PromoteUpperTo(d16, v13),
                hn::PromoteUpperTo(d16, v21), hn::PromoteUpperTo(d16, v23),
                hn::PromoteUpperTo(d16, v31), hn::PromoteUpperTo(d16, v32), hn::PromoteUpperTo(d16, v33));

            // Final store: 16-bit to 8-bit demotion
            auto final_val = hn::OrderedDemote2To(d8, res_lo, res_hi);
            hn::StoreU(final_val, d8, out + x);
        }
    }
}
void SobelKernel(const uint8_t* HWY_RESTRICT in, uint8_t* HWY_RESTRICT gradient, 
                 int width, int height, uint8_t threshold) {
    // We target 4 pixels at a time as our base 'Scalable' unit.
    // This allows easy promotion from 8 -> 16 -> 32 bit while keeping lane counts identical.
    const hn::FixedTag<uint8_t, 4> d8;
    const hn::FixedTag<int16_t, 4> d16;
    const hn::FixedTag<int32_t, 4> d32;

    const auto vDiv = hn::Set(d32, 9);
    const auto vThreshSq = hn::Set(d32, (int32_t)threshold * threshold);
    const auto v255 = hn::Set(d32, 255);
    const auto v0 = hn::Zero(d32);

    for (int y = 1; y < height - 1; ++y) {
        const uint8_t* r0 = in + (y - 1) * width;
        const uint8_t* r1 = in + y * width;
        const uint8_t* r2 = in + (y + 1) * width;
        uint8_t* out = gradient + y * width + 1;

        for (int x = 0; x < width; x += 4) {
            // Load and promote immediately to 32-bit to match naive 'int' math
            auto load32 = [&](const uint8_t* p) {
                return hn::PromoteTo(d32, hn::PromoteTo(d16, hn::LoadU(d8, p)));
            };

            auto p11 = load32(r0 + x);     auto p12 = load32(r0 + x + 1); auto p13 = load32(r0 + x + 2);
            auto p21 = load32(r1 + x);                                    auto p23 = load32(r1 + x + 2);
            auto p31 = load32(r2 + x);     auto p32 = load32(r2 + x + 1); auto p33 = load32(r2 + x + 2);

            // Note:: Division is very slow - we use it for now to match exactly with the naive non simd-implementation
            // sx = (*p11 + *p31 + 2 * *p21 - *p13 - 2 * *p23 - *p33) / 9;
            auto sx = hn::Div(hn::Sub(hn::Add(hn::Add(p11, p31), hn::Add(p21, p21)),
                                      hn::Add(hn::Add(p13, p33), hn::Add(p23, p23))), vDiv);
            
            // sy = (*p11 + *p13 + 2 * *p12 - *p31 - 2 * *p32 - *p33) / 9;
            auto sy = hn::Div(hn::Sub(hn::Add(hn::Add(p11, p13), hn::Add(p12, p12)),
                                      hn::Add(hn::Add(p31, p33), hn::Add(p32, p32))), vDiv);

            // int val = sx * sx + sy * sy;
            auto magSq = hn::Add(hn::Mul(sx, sx), hn::Mul(sy, sy));

            // *optr = val > thresholdSq ? 255 : 0;
            auto mask = hn::Gt(magSq, vThreshSq);
            auto res32 = hn::IfThenElse(mask, v255, v0);
            
            // Demote 32 -> 16 -> 8
            auto res8 = hn::DemoteTo(d8, hn::DemoteTo(d16, res32));
            hn::StoreU(res8, d8, out + x);
        }
    }
}
void SobelKerneli(const uint8_t* HWY_RESTRICT in, uint8_t* HWY_RESTRICT gradient, 
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

            // Process Upper Half 
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
