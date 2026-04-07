#include <benchmark/benchmark.h>
#include "gpc/forest.hpp"
#include "gpc/inference.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>

#define NUM_ELEMENTS 262668 //10*1224*375 //1024*1024

/**
 * Generates a reproducible Pareto-distributed vector.
 * @param count Number of IDs to generate.
 * @param target_mean The theoretical mean (requires alpha > 1).
 * @param seed A fixed value (e.g., 42) for deterministic benchmarks.
 */
std::vector<ndb::Descriptor> generate_pareto_ids(size_t count, double target_mean, uint32_t seed = 42) {
    std::vector<ndb::Descriptor> ids;
    ids.reserve(count);

    // Using a fixed seed for benchmark consistency
    std::mt19937 gen(seed); 
    
    // 1e-9 epsilon prevents division by zero/infinity
    std::uniform_real_distribution<double> dist(1e-9, 1.0);

    // Alpha = 1.16 provides a classic "80/20" Pareto distribution
    const double alpha = 1.16; 
    const double xm = target_mean * (alpha - 1.0) / alpha;

    for (size_t i = 0; i < count; ++i) {
        // Inverse Transform Sampling
        double val = xm / std::pow(dist(gen), 1.0 / alpha);
        
        // Casting to uint32_t will handle the Pareto "tail" by wrapping 
        // values that exceed 2^32-1, simulating a dense ID space.
        ids.push_back(ndb::Descriptor(ndb::Point(0,0), static_cast<uint32_t>(val)));
    }

    return ids;
}
std::vector<ndb::Descriptor> getSrcDescriptors() {
    std::vector<ndb::Descriptor> v =  ndb::Descriptor::deserialize("statesSrcLarge.txt", true);
    std::vector<ndb::Descriptor> out;
    for (size_t i = 0; i < v.size(); i++) {
        if (v[i].point.y % 5 == 0 && (v[i].state & 0xFFFFFFFF) != 0) { 
            out.push_back(v[i]);
        }
    }
    return out;
    //return generate_pareto_ids(NUM_ELEMENTS, 1000.0, 42); // 1M IDs with mean ~1000
}

std::vector<ndb::Descriptor> getTarDescriptors() {
    std::vector<ndb::Descriptor> v = ndb::Descriptor::deserialize("statesTarLarge.txt", false);
    std::vector<ndb::Descriptor> out;
    for (size_t i = 0; i < v.size(); i++) {
        if (v[i].point.y % 5 == 0 && (v[i].state & 0xFFFFFFFF) != 0) { 
            out.push_back(v[i]);
        }
    }
    return out;

    //return generate_pareto_ids(NUM_ELEMENTS, 1001.0, 42); // 1M IDs with mean ~1000
}
static void matchBySorting(
        benchmark::State& state) {
    std::vector<ndb::Descriptor> srcOriginal = getSrcDescriptors(); 
    std::vector<ndb::Descriptor> tarOriginal = getTarDescriptors();
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<ndb::Descriptor> src = srcOriginal;
        std::vector<ndb::Descriptor> tar = tarOriginal;
        state.ResumeTiming();
        std::vector<ndb::Correspondence> 
            matches = gpc::inference::Forest::findCorrespondences(src, tar);

        state.counters["matches"] = matches.size();
        benchmark::DoNotOptimize(matches);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(matchBySorting)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
