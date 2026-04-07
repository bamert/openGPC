#ifndef __NDB__KERNEL_GPC_HWY
#define __NDB__KERNEL_GPC_HWY

#include <hwy/highway.h>

#include <cstdint>

namespace ndb {

namespace testing {
void gpc_hwy(uint8_t* in,
             uint8_t* grad,
             uint32_t* HWY_RESTRICT gpc,
             const std::vector<int32_t>& fastmask,
             const std::vector<int32_t>& tau,
             int width,
             int height);

}

}  // namespace ndb

#endif  // GPC_KERNELS_SOBEL_HWY_H_
