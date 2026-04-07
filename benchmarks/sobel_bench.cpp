#include <benchmark/benchmark.h>
#include <hwy/highway.h>
#include "gpc/kernels/sobel.hpp" 
#include "gpc/kernels/sobel_hwy.hpp" 
static void BM_SobelHighway(benchmark::State& state) {
    int w = 1920, h = 1080;
    std::vector<uint8_t> in(w * h, 128);
    std::vector<uint8_t> out(w * h, 0);
    state.SetLabel(hwy::TargetName(HWY_TARGET));    
    // Warmup is handled automatically by the library
    for (auto _ : state) {
        ndb::testing::sobel_hwy(in.data(), out.data(), w, h, 50);
        
        // Ensure the compiler doesn't skip the work
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}

#if HWY_TARGET == HWY_AVX2
static void BM_SobelLegacySIMD(benchmark::State& state) {
    int w = 1920, h = 1080;
    std::vector<uint8_t> in(w * h, 128);
    std::vector<uint8_t> out(w * h, 0);

    state.SetLabel("AVX2_legacy");    
    for (auto _ : state) {
        ndb::sobelSSE(in.data(), out.data(), w, 1, h - 1, 1);
        
        // Ensure the compiler doesn't skip the work
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}
#endif
static void BM_SobelNaive(benchmark::State& state) {
    int w = 1920, h = 1080;
    std::vector<uint8_t> in(w * h, 128);
    std::vector<uint8_t> out(w * h, 0);

    state.SetLabel("naive");    
    for (auto _ : state) {
        ndb::sobelNaive(in.data(), out.data(), w, h, 50);
        
        // Ensure the compiler doesn't skip the work
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_SobelHighway)->Unit(benchmark::kMillisecond);
#if HWY_TARGET == HWY_AVX2
BENCHMARK(BM_SobelLegacySIMD)->Unit(benchmark::kMillisecond);
#endif
BENCHMARK(BM_SobelNaive)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
