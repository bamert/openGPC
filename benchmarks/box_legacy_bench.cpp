#include <benchmark/benchmark.h>
#include "gpc/filter.hpp" 

static void BM_BoxHighway(benchmark::State& state) {
    int w = 1920, h = 1080;
    std::vector<uint8_t> in(w * h, 128);
    std::vector<uint8_t> out(w * h, 0);

    for (auto _ : state) {
        ndb::box(in.data(), out.data(), w, h, 50);
        
        // Ensure the compiler doesn't skip the work
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(int64_t(state.iterations()) * w * h);
}
BENCHMARK(BM_BoxHighway)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
