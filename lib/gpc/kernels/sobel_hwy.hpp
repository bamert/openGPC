#ifndef  __NDB__KERNEL_SOBEL_HWY
#define __NDB__KERNEL_SOBEL_HWY

#include <cstdint>

namespace ndb {

namespace testing {
    /**
     * Entry point for benchmarking the MulHigh (approximate) version.
     */
    void sobel_hwy(uint8_t* in, uint8_t* blurred, int width, int height, uint8_t threshold);
}

}  // namespace ndb

#endif  // GPC_KERNELS_SOBEL_HWY_H_
