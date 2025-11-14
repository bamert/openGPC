add_executable(extract extract.cpp)
target_link_libraries(extract ${PNG_LIBRARIES} Threads::Threads)

add_executable(train train.cpp)
target_link_libraries(train ${PNG_LIBRARIES} Threads::Threads)

add_executable(sparsematch sparsematch.cpp)
target_link_libraries(sparsematch ${PNG_LIBRARIES} Threads::Threads)


