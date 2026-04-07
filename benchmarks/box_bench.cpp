#include <benchmark/benchmark.h>
#include <hwy/highway.h>
#include "gpc/kernels/box.hpp" 
#include "gpc/kernels/box_hwy.hpp" 
static void BM_BoxHighway(benchmark::State& state) {
    int w = 1920, h = 1080;
    std::vector<uint8_t> in(w * h, 128);
    std::vector<uint8_t> out(w * h, 0);
    state.SetLabel(hwy::TargetName(HWY_TARGET));    
    for (auto _ : state) {
        ndb::testing::box_hwy(in.data(), out.data(), w, h);
        
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}

#if HWY_TARGET == HWY_AVX2
static void BM_BoxLegacySIMD(benchmark::State& state) {
    int w = 1920, h = 1080;
    std::vector<uint8_t> in(w * h, 128);
    std::vector<uint8_t> out(w * h, 0);

    state.SetLabel("AVX2_legacy");    
    for (auto _ : state) {
        ndb::boxSSE(in.data(), out.data(), w, h);
        
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}
#endif
static void BM_BoxNaive(benchmark::State& state) {
    int w = 1920, h = 1080;
    std::vector<uint8_t> in(w * h, 128);
    std::vector<uint8_t> out(w * h, 0);

    state.SetLabel("naive");    
    for (auto _ : state) {
        ndb::boxNaive(in.data(), out.data(), w, h);
        
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_BoxHighway)->Unit(benchmark::kMillisecond);
#if HWY_TARGET == HWY_AVX2
BENCHMARK(BM_BoxLegacySIMD)->Unit(benchmark::kMillisecond);
#endif
BENCHMARK(BM_BoxNaive)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
