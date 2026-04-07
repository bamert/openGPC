// #define HWY_TARGET HWY_NEON
#include "gpc_hwy.hpp"
HWY_BEFORE_NAMESPACE();
namespace ndb {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// dense!
#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;

// Dense Version
void GPCKernel(const uint8_t* HWY_RESTRICT in,
               const uint8_t* HWY_RESTRICT grad,
               uint32_t* HWY_RESTRICT gpc,
               const std::vector<int32_t>& fastmask,
               const std::vector<int32_t>& tau,
               int width,
               int height) {
    const hn::ScalableTag<uint8_t> d8;
    const hn::ScalableTag<uint32_t> d32;
    const size_t N = hn::Lanes(d8);

    const int border = 13;
    const auto v_zero8 = hn::Zero(d8);
    const auto v_one8 = hn::Set(d8, 1);
    const int32_t* fm = fastmask.data();

    for (int y = border; y < height - border; ++y) {
        const int row_base = y * width;
        uint32_t* HWY_RESTRICT row_out = gpc + row_base;

        for (int x = border; x <= width - border - (int)N; x += N) {
            const int k = row_base + x;

            auto v_acc0 = hn::Zero(d8);  // Bits 0-7
            auto v_acc1 = hn::Zero(d8);  // Bits 8-15
            auto v_acc2 = hn::Zero(d8);  // Bits 16-23
            auto v_acc3 = hn::Zero(d8);  // Bits 24-31

            // Pass 1: Bits 0-7
            for (int i = 0; i < 16; i += 2) {
                v_acc0 = hn::Add(v_acc0, v_acc0);
                auto mask = hn::Gt(hn::LoadU(d8, in + k + fm[i]),
                                   hn::LoadU(d8, in + k + fm[i + 1]));
                v_acc0 = hn::Or(v_acc0, hn::IfThenElse(mask, v_one8, v_zero8));
            }

            // Pass 2: Bits 8-15
            for (int i = 16; i < 32; i += 2) {
                v_acc1 = hn::Add(v_acc1, v_acc1);
                auto mask = hn::Gt(hn::LoadU(d8, in + k + fm[i]),
                                   hn::LoadU(d8, in + k + fm[i + 1]));
                v_acc1 = hn::Or(v_acc1, hn::IfThenElse(mask, v_one8, v_zero8));
            }

            // Pass 3: Bits 16-23
            for (int i = 32; i < 48; i += 2) {
                v_acc2 = hn::Add(v_acc2, v_acc2);
                auto mask = hn::Gt(hn::LoadU(d8, in + k + fm[i]),
                                   hn::LoadU(d8, in + k + fm[i + 1]));
                v_acc2 = hn::Or(v_acc2, hn::IfThenElse(mask, v_one8, v_zero8));
            }

            // Pass 4: Bits 24-31
            for (int i = 48; i < 64; i += 2) {
                v_acc3 = hn::Add(v_acc3, v_acc3);
                auto mask = hn::Gt(hn::LoadU(d8, in + k + fm[i]),
                                   hn::LoadU(d8, in + k + fm[i + 1]));
                v_acc3 = hn::Or(v_acc3, hn::IfThenElse(mask, v_one8, v_zero8));
            }

            //extract and combine:
            for (size_t lane = 0; lane < N; ++lane) {
                uint32_t final_val =
                    (uint32_t(hn::ExtractLane(v_acc0, lane)) << 24) |
                    (uint32_t(hn::ExtractLane(v_acc1, lane)) << 16) |
                    (uint32_t(hn::ExtractLane(v_acc2, lane)) << 8) |
                    (uint32_t(hn::ExtractLane(v_acc3, lane)));
                row_out[x + lane] = final_val;
            }
        }
    }
}
void GPCKerneli(const uint8_t* HWY_RESTRICT in,
                const uint8_t* HWY_RESTRICT grad,
                uint32_t* HWY_RESTRICT gpc,
                const std::vector<int32_t>& fastmask,
                const std::vector<int32_t>& tau,
                int width,
                int height) {
    // We use the ScalableTag, but we will "Narrow" our view manually
    const hn::ScalableTag<uint32_t> d32;
    const hn::Rebind<uint8_t, decltype(d32)>
        d8_n;  // Same number of lanes as d32

    const size_t N = hn::Lanes(d32);
    const auto v_zero = hn::Zero(d32);
    const bool use_tau = !tau.empty();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; x += N) {
            const uint8_t* centerGrad = grad + y * width + x;

            // Load the gradient bytes for the current N lanes
            auto v_grad = hn::LoadU(d8_n, centerGrad);

            // Promotion-free zero check
            if (hn::AllTrue(d8_n, hn::Eq(v_grad, hn::Zero(d8_n)))) {
                continue;
            }

            auto v_tmp = hn::Zero(d32);

            for (size_t i = 0; i < fastmask.size(); i += 2) {
                v_tmp = hn::ShiftLeft<1>(v_tmp);

                // Promote N lanes of uint8 to N lanes of uint32
                auto v1 = hn::PromoteTo(
                    d32, hn::LoadU(d8_n, in + y * width + x + fastmask[i]));
                auto v2 = hn::PromoteTo(
                    d32, hn::LoadU(d8_n, in + y * width + x + fastmask[i + 1]));

                hn::Mask<decltype(d32)> mask;
                if (use_tau) {
                    auto v_tau = hn::Set(d32, tau[i / 2]);
                    mask = hn::Gt(v1, hn::Sub(v2, v_tau));
                } else {
                    mask = hn::Gt(v1, v2);
                }

                v_tmp = hn::Add(v_tmp,
                                hn::IfThenElse(mask, hn::Set(d32, 1), v_zero));
            }

            hn::StoreU(v_tmp, d32, gpc + y * width + x);
        }
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace ndb
HWY_AFTER_NAMESPACE();

namespace ndb {
namespace testing {
void gpc_hwy(uint8_t* in,
             uint8_t* grad,
             uint32_t* HWY_RESTRICT gpc,
             const std::vector<int32_t>& fastmask,
             const std::vector<int32_t>& tau,
             int width,
             int height) {
    HWY_STATIC_DISPATCH(GPCKernel)(in, grad, gpc, fastmask, tau, width, height);
}
}  // namespace testing
}  // namespace ndb
