cmake_minimum_required(VERSION 3.10)
include(CheckCXXCompilerFlag)
include(CheckCXXSourceRuns)
project(openGPC CXX)
set (REQ_CPP11_FEATURES  cxx_strong_enums cxx_auto_type)

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

add_compile_options(-O3 -funroll-loops)
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

add_library(gpc_core 
    lib/gpc/forest.cpp 
)
target_link_libraries(gpc_core 
    PUBLIC 
        Eigen3::Eigen 
        ${PNG_LIBRARIES} 
        Threads::Threads
)
target_include_directories(gpc_core PUBLIC lib)
enable_testing()
add_subdirectory(samples)
add_subdirectory(tests)

