#include <benchmark/benchmark.h>
#include "gpc/inference.hpp"

typedef gpc::inference::Forest GPCForest_t;
GPCForest_t forest;

static void fullInference(
        benchmark::State& state){

    std::string forestPath = "../forests/defaultZeroForest.txt";
    std::string leftImgPath = "../data/middlebury/im0.png";
    std::string rightImgPath = "../data/middlebury/im1.png";
    gpc::inference::InferenceSettings inferencesettings =
        gpc::inference::InferenceSettings()
            .builder()
            .gradientThreshold(state.range(0)) // 0...255 gradient threshold for sobel filter
            .verticalTolerance(
                0)               // 0px tolerance for rectified epipolar matches
            .dispHigh(128)       // limit disparities to 128
            .epipolarMode(true)  // match GPC states in epipolar mode. more
                                 // matches, lower accuracy than global
            .useHashtable(false);  // use sort method for matching. faster for
                                   // <100K descriptors

    ndb::Buffer<uint8_t> simg, timg;
    // Load images
    simg.readPNG(leftImgPath);
    timg.readPNG(rightImgPath);

    // Get learned filter for the given image dimensions.
    GPCForest_t::FilterMask fm =
        forest.readForest(forestPath, simg.cols(), simg.rows());



    for (auto _ : state) {
        GPCForest_t::PreprocessedImage simgP =
            forest.preprocessImage(simg, inferencesettings);
        GPCForest_t::PreprocessedImage timgP =
            forest.preprocessImage(timg, inferencesettings);
        std::vector<ndb::Support> supp =
            forest.rectifiedMatch(simgP, timgP, fm, inferencesettings);
        state.counters["f_s"] = simgP.mask.size();
        state.counters["f_t"] = timgP.mask.size();
        state.counters["matches"] = supp.size();
        benchmark::DoNotOptimize(supp);
        benchmark::ClobberMemory();
    }
 
}

BENCHMARK(fullInference)
    ->Unit(benchmark::kMillisecond)
    ->Args({0})
    ->Args({5})
    ->Args({100});


BENCHMARK_MAIN();
/*
int main(int argc, char** argv) {

        BenchmarkResults b = fullInference(simg,timg, fm, inferenceSettings);
    for (const auto& [name, time] : b) {
        cout << name << ", " << time << " ms" << endl;
    }

}
*/
