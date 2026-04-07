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
add_executable(box_bench box_bench.cpp)

target_link_libraries(box_bench
    PRIVATE 
        gpc_core
        benchmark::benchmark 
)
add_executable(correspondence_bench correspondence_bench.cpp)

target_link_libraries(correspondence_bench
    PRIVATE 
        gpc_core
        benchmark::benchmark 
)
