add_executable(sobel_bench sobel_bench.cpp)

target_link_libraries(sobel_bench
    PRIVATE 
        gpc_core 
        benchmark::benchmark 
        hwy
)

set_target_properties(sobel_bench PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

add_executable(kernel_bench kernel_bench.cpp)
target_link_libraries(kernel_bench 
    PRIVATE 
        gpc_core
        benchmark::benchmark 
)
add_executable(sobel_legacy_bench sobel_legacy_bench.cpp)

target_link_libraries(sobel_legacy_bench
    PRIVATE 
        gpc_core
        benchmark::benchmark 
)
add_executable(box_legacy_bench box_legacy_bench.cpp)

target_link_libraries(box_legacy_bench
    PRIVATE 
        gpc_core
        benchmark::benchmark 
)
