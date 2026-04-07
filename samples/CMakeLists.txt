add_executable(extract extract.cpp)
target_link_libraries(extract gpc_core)

add_executable(train train.cpp)
target_link_libraries(train gpc_core)

add_executable(sparsematch sparsematch.cpp)
target_link_libraries(sparsematch gpc_core)

add_executable(target target.cpp)
target_link_libraries(target gpc_core)


