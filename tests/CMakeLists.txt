FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.4
)
FetchContent_MakeAvailable(Catch2)

add_executable(test_dexel test_main.cpp)
target_link_libraries(test_dexel PRIVATE dexel Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(test_dexel)
