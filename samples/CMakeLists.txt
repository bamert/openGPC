cmake_minimum_required(VERSION 3.4)

project(openGPC CXX)
# Use C++11 features.
set (REQ_CPP11_FEATURES  cxx_strong_enums cxx_auto_type)

set(CMAKE_CXX_STANDARD 11) # C++11...
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

#By default, use SSE intrinsics
option(SSE "SSE" ON)
if(SSE)
  add_definitions(-D_INTRINSICS_SSE)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3  -mavx2 -march=core-avx2 -funroll-loops")
else()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3  -funroll-loops")
endif()

#find pnglib (used to load and store images for training and during evaluation)
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})


#Add gpc header directy
include_directories("../lib")

#Three examples for GPC library
add_executable(extract extract.cpp)
target_link_libraries(extract ${PNG_LIBRARIES})

add_executable(train train.cpp)
target_link_libraries(train ${PNG_LIBRARIES})


add_executable(sparsematch sparsematch.cpp)
target_link_libraries(sparsematch ${PNG_LIBRARIES})


