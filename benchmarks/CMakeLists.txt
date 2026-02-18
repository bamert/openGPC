add_executable(kernel_bench kernel_bench.cpp)

target_link_libraries(kernel_bench 
    PRIVATE 
        benchmark::benchmark 
        Eigen3::Eigen
        ${PNG_LIBRARIES} 
)
