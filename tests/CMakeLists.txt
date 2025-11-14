find_package(GTest REQUIRED)

include(FetchContent)

FetchContent_Declare(
  approvaltests
  GIT_REPOSITORY https://github.com/approvals/ApprovalTests.cpp.git
  GIT_TAG v.10.12.0       
)
FetchContent_MakeAvailable(approvaltests)

enable_testing()

find_package(GTest REQUIRED)
add_executable(test_single_matching test_single_matching.cpp)
target_link_libraries(test_single_matching PRIVATE ${PNG_LIBRARIES} ApprovalTests::ApprovalTests GTest::gtest_main)

add_test(NAME single_matching COMMAND test_single_matching)

