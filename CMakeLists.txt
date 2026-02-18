cmake_minimum_required(VERSION 3.10)
include(CheckCXXCompilerFlag)
include(CheckCXXSourceRuns)
project(openGPC CXX)
set (REQ_CPP11_FEATURES  cxx_strong_enums cxx_auto_type)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
find_package(PNG REQUIRED)
find_package(Threads REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PNG_INCLUDE_DIRS})
include_directories(lib)

#By default, use SSE intrinsics
option(SSE "Enable SSE/AVX optimizations if available" ON)

add_compile_options(-O3 -funroll-loops -flto)
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    add_compile_options(-mcpu=apple-m1)
elseif(NOT MSVC)
    add_compile_options(-march=native)
endif()
if(SSE)
    message(STATUS "Checking if target CPU supports AVX2 instructions...")
    check_cxx_source_runs("
      #include <immintrin.h>
      int main() {
        __m256i x = _mm256_set1_epi32(1);
        return _mm256_extract_epi32(x, 0);
      }
    " CPU_HAS_AVX2)

    if(CPU_HAS_AVX2)
      message(STATUS "AVX2: supported and enabled")
      add_compile_definitions(_INTRINSICS_SSE)
      add_compile_options(-mavx2 -march=core-avx2)
    endif()
endif()
include(FetchContent)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "Disable Highway tests" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "Disable Highway examples" FORCE)
FetchContent_Declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG        1.3.0  
)
FetchContent_MakeAvailable(highway)

include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_LIB OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  google_benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.9.5  
)
FetchContent_MakeAvailable(google_benchmark)
add_library(gpc_core 
    lib/gpc/forest.cpp 
    lib/gpc/fern.cpp 
    lib/gpc/feature.cpp 
    lib/gpc/kernels/sobel.cpp
    lib/gpc/kernels/box.cpp
    lib/gpc/kernels/census.cpp
    lib/gpc/kernels/gpc.cpp
    lib/gpc/kernels/utils.cpp
    lib/gpc/kernels/box_hwy.cpp
    lib/gpc/kernels/sobel_hwy.cpp
)
target_link_libraries(gpc_core 
    PUBLIC 
        Eigen3::Eigen 
        ${PNG_LIBRARIES} 
        Threads::Threads
        hwy
)
target_include_directories(gpc_core PUBLIC lib)
enable_testing()
add_subdirectory(samples)
add_subdirectory(tests)
add_subdirectory(benchmarks)

