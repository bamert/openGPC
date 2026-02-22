#ifndef  __NDB__KERNEL_BOX_HWY
#define __NDB__KERNEL_BOX_HWY

#include <cstdint>

namespace ndb {

namespace testing {
    /**
     * Entry point for benchmarking the MulHigh (approximate) version.
     */
    void box_hwy(uint8_t* in, uint8_t* blurred, int width, int height);

}

}  // namespace ndb

#endif  // GPC_KERNELS_BOX_HWY_H_
