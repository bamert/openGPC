#include <hwy/highway.h>
#include <iostream>
int main() {
    // This is evaluated at compile-time
    std::cout << "Compiled for: " << hwy::TargetName(HWY_TARGET) << std::endl;

    // If you need logic based on the arch:
#if HWY_TARGET == HWY_AVX2
    std::cout << "Logic: Using 256-bit AVX2 paths." << std::endl;
#elif HWY_TARGET == HWY_NEON
    std::cout << "Logic: Using 128-bit NEON paths." << std::endl;
#elif HWY_TARGET == HWY_SSE4
    std::cout << "Logic: Using 128-bit SSE4 paths." << std::endl;
#else
    std::cout << "Logic: Using Scalar fallback." << std::endl;
#endif
}
