cmake_minimum_required(VERSION 3.14)

project(my_project LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

# fetch dependencies
FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.3.2
)
FetchContent_MakeAvailable(CLI11)

FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(Eigen)

FetchContent_Declare(
    Libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG v2.4.0
)
FetchContent_MakeAvailable(Libigl)

# ライブラリとアプリをビルド
add_subdirectory(src)
add_subdirectory(app)

enable_testing()
add_subdirectory(tests)
