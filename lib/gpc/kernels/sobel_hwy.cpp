#define HWY_TARGET HWY_NEON 
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE(); 
namespace ndb {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;


void SobelKernel(const uint8_t* HWY_RESTRICT in, uint8_t* HWY_RESTRICT gradient, 
                 int width, int height, uint8_t threshold) {
    const hn::ScalableTag<uint8_t> d8;
    const hn::Rebind<int16_t, hn::Half<decltype(d8)>> d16; // Signed 16-bit, half the lanes of d8
    const hn::Half<decltype(d8)> d8_half; // Tag for half-width 8-bit loads
    
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
            // Load full 128-bit vectors
            auto v11 = hn::LoadU(d8, r0 + x); auto v12 = hn::LoadU(d8, r0 + x + 1); auto v13 = hn::LoadU(d8, r0 + x + 2);
            auto v21 = hn::LoadU(d8, r1 + x);                                      auto v23 = hn::LoadU(d8, r1 + x + 2);
            auto v31 = hn::LoadU(d8, r2 + x); auto v32 = hn::LoadU(d8, r2 + x + 1); auto v33 = hn::LoadU(d8, r2 + x + 2);

            // LOWER HALF PROCESSING
            {
                // PromoteTo signed 16-bit from the lower half of our 8-bit vectors
                auto p11 = hn::PromoteTo(d16, hn::LowerHalf(v11));
                auto p12 = hn::PromoteTo(d16, hn::LowerHalf(v12));
                auto p13 = hn::PromoteTo(d16, hn::LowerHalf(v13));
                auto p21 = hn::PromoteTo(d16, hn::LowerHalf(v21));
                auto p23 = hn::PromoteTo(d16, hn::LowerHalf(v23));
                auto p31 = hn::PromoteTo(d16, hn::LowerHalf(v31));
                auto p32 = hn::PromoteTo(d16, hn::LowerHalf(v32));
                auto p33 = hn::PromoteTo(d16, hn::LowerHalf(v33));

                auto sx = hn::Sub(hn::Add(hn::Add(p11, p31), hn::Add(p21, p21)), 
                                  hn::Add(hn::Add(p13, p33), hn::Add(p23, p23)));
                sx = hn::MulHigh(sx, divisor);

                auto sy = hn::Sub(hn::Add(hn::Add(p11, p13), hn::Add(p12, p12)), 
                                  hn::Add(hn::Add(p31, p33), hn::Add(p32, p32)));
                sy = hn::MulHigh(sy, divisor);

                auto mag = hn::Add(hn::Mul(sx, sx), hn::Mul(sy, sy));
                auto mask = hn::Gt(mag, threshSq);
                auto res_lo = hn::DemoteTo(d8_half, hn::IfThenElse(mask, v255, v0));

                // UPPER HALF PROCESSING
                auto u11 = hn::PromoteTo(d16, hn::UpperHalf(d8, v11));
                auto u12 = hn::PromoteTo(d16, hn::UpperHalf(d8, v12));
                auto u13 = hn::PromoteTo(d16, hn::UpperHalf(d8, v13));
                auto u21 = hn::PromoteTo(d16, hn::UpperHalf(d8, v21));
                auto u23 = hn::PromoteTo(d16, hn::UpperHalf(d8, v23));
                auto u31 = hn::PromoteTo(d16, hn::UpperHalf(d8, v31));
                auto u32 = hn::PromoteTo(d16, hn::UpperHalf(d8, v32));
                auto u33 = hn::PromoteTo(d16, hn::UpperHalf(d8, v33));

                auto sx_u = hn::Sub(hn::Add(hn::Add(u11, u31), hn::Add(u21, u21)), 
                                    hn::Add(hn::Add(u13, u33), hn::Add(u23, u23)));
                sx_u = hn::MulHigh(sx_u, divisor);

                auto sy_u = hn::Sub(hn::Add(hn::Add(u11, u13), hn::Add(u12, u12)), 
                                    hn::Add(hn::Add(u31, u33), hn::Add(u32, u32)));
                sy_u = hn::MulHigh(sy_u, divisor);

                auto mag_u = hn::Add(hn::Mul(sx_u, sx_u), hn::Mul(sy_u, sy_u));
                auto mask_u = hn::Gt(mag_u, threshSq);
                auto res_hi = hn::DemoteTo(d8_half, hn::IfThenElse(mask_u, v255, v0));

                hn::StoreU(hn::Combine(d8, res_hi, res_lo), d8, out + x);
            }
        }
    }
}
void SobelKerneli(const uint8_t* HWY_RESTRICT in, uint8_t* HWY_RESTRICT gradient, 
                 int width, int height, uint8_t threshold) {
    const hn::ScalableTag<uint8_t> d8;
    const hn::Rebind<int16_t, hn::Half<decltype(d8)>> d16; 
    const hn::Half<decltype(d8)> d8_half; 
    
    const size_t N = hn::Lanes(d8);
    // Multiply threshold by 9 BEFORE squaring to match the "no-division" math
    int16_t tScaled = (int16_t)threshold * 9;
    const auto threshSq = hn::Set(d16, tScaled * tScaled);
    
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

            // LOWER HALF
            {
                auto p11 = hn::PromoteTo(d16, hn::LowerHalf(v11));
                auto p12 = hn::PromoteTo(d16, hn::LowerHalf(v12));
                auto p13 = hn::PromoteTo(d16, hn::LowerHalf(v13));
                auto p21 = hn::PromoteTo(d16, hn::LowerHalf(v21));
                auto p23 = hn::PromoteTo(d16, hn::LowerHalf(v23));
                auto p31 = hn::PromoteTo(d16, hn::LowerHalf(v31));
                auto p32 = hn::PromoteTo(d16, hn::LowerHalf(v32));
                auto p33 = hn::PromoteTo(d16, hn::LowerHalf(v33));

                auto sx = hn::Sub(hn::Add(hn::Add(p11, p31), hn::Add(p21, p21)), 
                                  hn::Add(hn::Add(p13, p33), hn::Add(p23, p23)));
                auto sy = hn::Sub(hn::Add(hn::Add(p11, p13), hn::Add(p12, p12)), 
                                  hn::Add(hn::Add(p31, p33), hn::Add(p32, p32)));

                // Removed MulHigh (division). Math is now: (sx*sx + sy*sy) > (threshold*9)^2
                auto mag = hn::Add(hn::Mul(sx, sx), hn::Mul(sy, sy));
                auto mask = hn::Gt(mag, threshSq);
                auto res_lo = hn::DemoteTo(d8_half, hn::IfThenElse(mask, v255, v0));

                // UPPER HALF
                auto u11 = hn::PromoteTo(d16, hn::UpperHalf(d8, v11));
                auto u12 = hn::PromoteTo(d16, hn::UpperHalf(d8, v12));
                auto u13 = hn::PromoteTo(d16, hn::UpperHalf(d8, v13));
                auto u21 = hn::PromoteTo(d16, hn::UpperHalf(d8, v21));
                auto u23 = hn::PromoteTo(d16, hn::UpperHalf(d8, v23));
                auto u31 = hn::PromoteTo(d16, hn::UpperHalf(d8, v31));
                auto u32 = hn::PromoteTo(d16, hn::UpperHalf(d8, v32));
                auto u33 = hn::PromoteTo(d16, hn::UpperHalf(d8, v33));

                auto sx_u = hn::Sub(hn::Add(hn::Add(u11, u31), hn::Add(u21, u21)), 
                                    hn::Add(hn::Add(u13, u33), hn::Add(u23, u23)));
                auto sy_u = hn::Sub(hn::Add(hn::Add(u11, u13), hn::Add(u12, u12)), 
                                    hn::Add(hn::Add(u31, u33), hn::Add(u32, u32)));

                auto mag_u = hn::Add(hn::Mul(sx_u, sx_u), hn::Mul(sy_u, sy_u));
                auto mask_u = hn::Gt(mag_u, threshSq);
                auto res_hi = hn::DemoteTo(d8_half, hn::IfThenElse(mask_u, v255, v0));

                hn::StoreU(hn::Combine(d8, res_hi, res_lo), d8, out + x);
            }
        }
    }
}
} // namespace HWY_NAMESPACE
} // namespace ndb
HWY_AFTER_NAMESPACE();

namespace ndb {
namespace testing {
    void sobel_hwy(uint8_t* in, uint8_t* blurred, int width, int height, uint8_t threshold) {
        ndb::N_NEON::SobelKernel(in, blurred, width, height, threshold);
    }
}
}
