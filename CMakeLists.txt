cmake_minimum_required(VERSION 3.14)

project(my_project)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.3.2
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


add_executable(my_project src/main.cpp)
target_link_libraries(my_project PRIVATE CLI11 igl::core Eigen3::Eigen)

target_include_directories(my_project PRIVATE
    ${Eigen_SOURCE_DIR}
    ${Libigl_SOURCE_DIR}/include
)

