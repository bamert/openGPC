#include <hwy/highway.h>

#include <iostream>
int main() {
    std::cout << "Compiled for: " << hwy::TargetName(HWY_TARGET) << std::endl;
#if HWY_TARGET == HWY_AVX2
    std::cout << "Using 256-bit AVX2 paths." << std::endl;
#elif HWY_TARGET == HWY_NEON
    std::cout << "Using 128-bit NEON paths." << std::endl;
#elif HWY_TARGET == HWY_SSE4
    std::cout << "Using 128-bit SSE4 paths." << std::endl;
#else
    std::cout << "Using Scalar fallback." << std::endl;
#endif
}
