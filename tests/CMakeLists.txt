find_package(GTest REQUIRED)

include(FetchContent)

FetchContent_Declare(
  approvaltests
  GIT_REPOSITORY https://github.com/approvals/ApprovalTests.cpp.git
  GIT_TAG v.10.12.0       
)
FetchContent_MakeAvailable(approvaltests)


find_package(GTest REQUIRED)
function(add_gpc_approval_test TEST_NAME SOURCE_FILE)
    add_executable(${TEST_NAME} ${SOURCE_FILE})
    
    target_link_libraries(${TEST_NAME} 
        PRIVATE 
            gpc_core 
            ApprovalTests::ApprovalTests 
            GTest::gtest_main
    )
    

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

add_gpc_approval_test(test_single_matching test_single_matching.cpp)
add_gpc_approval_test(test_kernel_box test_kernel_box.cpp)
add_gpc_approval_test(test_kernel_sobel test_kernel_sobel.cpp)
add_gpc_approval_test(test_kernel_gpc test_kernel_gpc.cpp)

