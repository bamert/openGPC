add_executable(kernel_bench kernel_bench.cpp)

target_link_libraries(kernel_bench 
    PRIVATE 
        benchmark::benchmark 
        Eigen3::Eigen
        ${PNG_LIBRARIES} 
)
add_executable(sobel_legacy_bench sobel_legacy_bench.cpp)

target_link_libraries(sobel_legacy_bench
    PRIVATE 
        benchmark::benchmark 
        Eigen3::Eigen
        ${PNG_LIBRARIES} 
)
add_executable(box_legacy_bench box_legacy_bench.cpp)

target_link_libraries(box_legacy_bench
    PRIVATE 
        benchmark::benchmark 
        Eigen3::Eigen
        ${PNG_LIBRARIES} 
)
