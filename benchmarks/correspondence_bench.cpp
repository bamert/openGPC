#include <benchmark/benchmark.h>
#include "gpc/forest.hpp"
#include "gpc/inference.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>

#define NUM_ELEMENTS 262668 //10*1224*375 //1024*1024

/* Remaining ideas
 * -USE ILP: Parallel Radix Partitioning (Even on one core, using a single-pass shuffle).
 *      - Didn't speed up. was same as matchPreparedFramesFaster
 *      - Assuming that the bottleneck is the hash table probes, hence: look into bloom filters...
 * -Blocked Bloom Filter to discard non-matches in L1.
 *      - Faster at 1M, slower at 100K and 10M 
 * -SIMD-Probed Flat Table (checking 4 slots at once).
 * -Manual Prefetching of the next bucket's data.
 * */
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
    return ndb::Descriptor::deserialize("statesSrc.txt", true);
    //return generate_pareto_ids(NUM_ELEMENTS, 1000.0, 42); // 1M IDs with mean ~1000
}

std::vector<ndb::Descriptor> getTarDescriptors() {
    return ndb::Descriptor::deserialize("statesTar.txt", false);
    //return generate_pareto_ids(NUM_ELEMENTS, 1001.0, 42); // 1M IDs with mean ~1000
}
std::vector<ndb::Descriptor> generate_unique_ids(size_t count) {
    std::vector<ndb::Descriptor> ids;
    ids.reserve(count);


    for (size_t i = 0; i < count; ++i) {
        ids.push_back(ndb::Descriptor(ndb::Point(0,0), static_cast<uint32_t>(i)));
    }
    return ids;
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
        //state.counters["candidates_t"] = timgP.mask.size();
        //state.counters["matches"] = supp.size();
        benchmark::DoNotOptimize(matches);
        benchmark::ClobberMemory();
    }
}
static void matchByHashingNaive(
        benchmark::State& state) {
    std::vector<ndb::Descriptor> srcOriginal = getSrcDescriptors(); 
    std::vector<ndb::Descriptor> tarOriginal = getTarDescriptors();
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<ndb::Descriptor> src = srcOriginal;
        std::vector<ndb::Descriptor> tar = tarOriginal;
        state.ResumeTiming();
        std::vector<ndb::Correspondence> 
            matches = gpc::inference::Forest::findCorrespondencesHashNaive(src, tar);

        state.counters["matches"] = matches.size();
        benchmark::DoNotOptimize(matches);
        benchmark::ClobberMemory();
    }
}
static void matchByHashing(
        benchmark::State& state) {
    std::vector<ndb::Descriptor> srcOriginal = getSrcDescriptors(); 
    std::vector<ndb::Descriptor> tarOriginal = getTarDescriptors();
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<ndb::Descriptor> src = srcOriginal;
        std::vector<ndb::Descriptor> tar = tarOriginal;
        state.ResumeTiming();
        std::vector<ndb::Correspondence> 
            matches = gpc::inference::Forest::findCorrespondencesTurbo(src, tar);

        state.counters["matches"] = matches.size();
        //state.counters["candidates_t"] = timgP.mask.size();
        //state.counters["matches"] = supp.size();
        benchmark::DoNotOptimize(matches);
        benchmark::ClobberMemory();
    }
}
static void matchPreparedFrames(
        benchmark::State& state) {
    std::vector<ndb::Descriptor> srcOriginal = getSrcDescriptors(); 
    std::vector<ndb::Descriptor> tarOriginal = getTarDescriptors();
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<ndb::Descriptor> src = srcOriginal;
        std::vector<ndb::Descriptor> tar = tarOriginal;
        state.ResumeTiming();
        auto v = gpc::inference::Forest::prepareSoAFrames(src, tar);
        auto matches = gpc::inference::Forest::matchPreparedFrames(v.first, v.second);

        state.counters["matches"] = matches.first.size();
        //state.counters["candidates_t"] = timgP.mask.size();
        //state.counters["matches"] = supp.size();
        benchmark::DoNotOptimize(matches);
        benchmark::ClobberMemory();
    }
}
static void matchPreparedFramesFaster(
        benchmark::State& state) {
    std::vector<ndb::Descriptor> srcOriginal = getSrcDescriptors(); 
    std::vector<ndb::Descriptor> tarOriginal = getTarDescriptors();
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<ndb::Descriptor> src = srcOriginal;
        std::vector<ndb::Descriptor> tar = tarOriginal;
        state.ResumeTiming();
        auto v = gpc::inference::Forest::prepareSoAFrames(src, tar);
        //auto matches = gpc::inference::Forest::matchPreparedFramesFaster(v.first, v.second);
        auto matches = gpc::inference::Forest::matchParallelRadixPartitioning(v.first,v.second);
        state.counters["matches"] = matches.first.size();
        //state.counters["candidates_t"] = timgP.mask.size();
        //state.counters["matches"] = supp.size();
        benchmark::DoNotOptimize(matches);
        benchmark::ClobberMemory();
    }
}
static void matchParallelRadixPartitioning(
        benchmark::State& state) {
    std::vector<ndb::Descriptor> srcOriginal = getSrcDescriptors(); 
    std::vector<ndb::Descriptor> tarOriginal = getTarDescriptors();
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<ndb::Descriptor> src = srcOriginal;
        std::vector<ndb::Descriptor> tar = tarOriginal;
        state.ResumeTiming();
        auto v = gpc::inference::Forest::prepareSoAFrames(src,tar);
        auto matches = gpc::inference::Forest::matchParallelRadixPartitioning(v.first, v.second);

        state.counters["matches"] = matches.first.size();
        //state.counters["candidates_t"] = timgP.mask.size();
        //state.counters["matches"] = supp.size();
        benchmark::DoNotOptimize(matches);
        benchmark::ClobberMemory();
    }
}
static void matchBlockedBloom(
        benchmark::State& state) {
    std::vector<ndb::Descriptor> srcOriginal = getSrcDescriptors(); 
    std::vector<ndb::Descriptor> tarOriginal = getTarDescriptors();
                                                    
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<ndb::Descriptor> src = srcOriginal;
        std::vector<ndb::Descriptor> tar = tarOriginal;
        state.ResumeTiming();
        auto v = gpc::inference::Forest::prepareSoAFrames(src,tar);
        auto matches = gpc::inference::Forest::matchBlockedBloom(v.first, v.second);

        state.counters["matches"] = matches.first.size();
        benchmark::DoNotOptimize(matches);
        benchmark::ClobberMemory();
    }
}
static void matchAdaptive(
        benchmark::State& state) {
    std::vector<ndb::Descriptor> srcOriginal = getSrcDescriptors(); 
    std::vector<ndb::Descriptor> tarOriginal = getTarDescriptors();
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<ndb::Descriptor> src = srcOriginal;
        std::vector<ndb::Descriptor> tar = tarOriginal;
        state.ResumeTiming();
        auto v = gpc::inference::Forest::prepareSoAFrames(src,tar);
        auto matches = gpc::inference::Forest::matchAdaptive(v.first, v.second);

        state.counters["matches"] = matches.first.size();
        benchmark::DoNotOptimize(matches);
        benchmark::ClobberMemory();
    }
}
/*
static void matchAdaptiveNeon(
        benchmark::State& state) {
    std::vector<ndb::Descriptor> src, tar;
    src = generate_pareto_ids(NUM_ELEMENTS, 1000.0, 42); // 1M IDs with mean ~1000
    tar = generate_pareto_ids(NUM_ELEMENTS, 1001.0, 42); // 1M IDs with mean ~1000
                                                    
    for (auto _ : state) {
        auto v = gpc::inference::Forest::prepareSoAFrames(src, tar);
        auto matches = gpc::inference::Forest::matchAdaptiveNeon(v.first, v.second);

        state.counters["matches"] = matches.first.size();
        //state.counters["candidates_t"] = timgP.mask.size();
        //state.counters["matches"] = supp.size();
        benchmark::DoNotOptimize(matches);
        benchmark::ClobberMemory();
    }
}
*/
static void matchAdaptivePersistent(
        benchmark::State& state) {
    std::vector<ndb::Descriptor> srcOriginal = getSrcDescriptors(); 
    std::vector<ndb::Descriptor> tarOriginal = getTarDescriptors();
    gpc::inference::SoAFramePersistent srcFrame, tarFrame;
    srcFrame.preallocate(srcOriginal.size()); // size known
    tarFrame.preallocate(tarOriginal.size());
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<ndb::Descriptor> src = srcOriginal;
        std::vector<ndb::Descriptor> tar = tarOriginal;
        state.ResumeTiming();

        gpc::inference::Forest::prepareSoAFramesPersistent(src, tar, srcFrame, tarFrame);
        auto matches = gpc::inference::Forest::matchAdaptivePersistent(srcFrame, tarFrame);

        state.counters["matches"] = matches.first.size();
        benchmark::DoNotOptimize(matches);
        benchmark::ClobberMemory();
    }
}
static void matchPipelinedBranchless(
        benchmark::State& state) {
    std::vector<ndb::Descriptor> srcOriginal = getSrcDescriptors(); 
    std::vector<ndb::Descriptor> tarOriginal = getTarDescriptors();
    gpc::inference::SoAFramePersistent srcFrame, tarFrame;
    srcFrame.preallocate(srcOriginal.size()); // size known
    tarFrame.preallocate(tarOriginal.size());
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<ndb::Descriptor> src = srcOriginal;
        std::vector<ndb::Descriptor> tar = tarOriginal;
        state.ResumeTiming();

        gpc::inference::Forest::prepareSoAFramesPersistent(src, tar, srcFrame, tarFrame);
        auto matches = gpc::inference::Forest::matchPipelinedBranchless(srcFrame, tarFrame);
        benchmark::DoNotOptimize(matches);
        benchmark::ClobberMemory();
    }
}
static void matchPipelinedBranchlessPreallocate(
        benchmark::State& state) {
    std::vector<ndb::Descriptor> srcOriginal = getSrcDescriptors(); 
    std::vector<ndb::Descriptor> tarOriginal = getTarDescriptors();
    gpc::inference::SoAFramePersistent srcFrame, tarFrame;
    srcFrame.preallocate(srcOriginal.size()); // size known
    tarFrame.preallocate(tarOriginal.size());
    std::vector<uint32_t> resultSrc, resultTar;
    resultSrc.reserve(srcOriginal.size()/100);
    resultTar.reserve(tarOriginal.size()/100);
    for (auto _ : state) {
        state.PauseTiming();
        resultSrc.clear();
        resultTar.clear();
        std::vector<ndb::Descriptor> src = srcOriginal;
        std::vector<ndb::Descriptor> tar = tarOriginal;
        state.ResumeTiming();

        // 1. Measure Prepare
        // 2M: 5.7ms, 20M: 57ms
        gpc::inference::Forest::prepareSoAFramesPersistent(src, tar, srcFrame, tarFrame);

        // 2. Measure Match
        // 2M: 5.3ms , 20M: 53ms
        gpc::inference::Forest::matchPipelinedBranchlessPreallocate(srcFrame, tarFrame, resultSrc, resultTar);

        state.counters["matches"] = resultSrc.size();
        
        benchmark::DoNotOptimize(resultSrc);
        benchmark::DoNotOptimize(resultTar);
        benchmark::ClobberMemory();
    }
}

static void matchPipelinedBranchlessPreallocateSingleSlab(
        benchmark::State& state) {
    std::vector<ndb::Descriptor> srcOriginal = getSrcDescriptors(); 
    std::vector<ndb::Descriptor> tarOriginal = getTarDescriptors();
                                                    
    gpc::inference::SoAFramePersistentSingleSlab srcFrame, tarFrame;
    srcFrame.preallocate(srcOriginal.size()); // size known
    tarFrame.preallocate(tarOriginal.size());
    std::vector<uint32_t> resultSrc, resultTar;
    resultSrc.reserve(srcOriginal.size()/10);
    resultTar.reserve(tarOriginal.size()/10);
    for (auto _ : state) {
        state.PauseTiming();
        resultSrc.clear();
        resultTar.clear();
        // 1. Measure Prepare
        // 2M: 5.7ms, 20M: 57ms
        std::vector<ndb::Descriptor> src = srcOriginal;
        std::vector<ndb::Descriptor> tar = tarOriginal;
        state.ResumeTiming();
        gpc::inference::Forest::prepareSoAFramesPersistentSingleSlab(src, tar, srcFrame, tarFrame);

        // 2. Measure Match
        // 2M: 5.3ms , 20M: 53ms
        gpc::inference::Forest::matchPipelinedBranchlessPreallocateSingleSlab(srcFrame, tarFrame, resultSrc, resultTar);

        state.counters["matches"] = resultSrc.size();
        
        benchmark::DoNotOptimize(resultSrc);
        benchmark::DoNotOptimize(resultTar);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(matchBySorting)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(matchByHashingNaive)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(matchByHashing)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(matchPreparedFrames)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(matchPreparedFramesFaster)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(matchParallelRadixPartitioning)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(matchBlockedBloom)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(matchAdaptive)
    ->Unit(benchmark::kMillisecond);
/*
BENCHMARK(matchAdaptiveNeon)
    ->Unit(benchmark::kMillisecond);
*/
BENCHMARK(matchAdaptivePersistent)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(matchPipelinedBranchless)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(matchPipelinedBranchlessPreallocate)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(matchPipelinedBranchlessPreallocateSingleSlab)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
