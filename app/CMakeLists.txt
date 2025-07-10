add_executable(my_project main.cpp)

target_link_libraries(my_project
    PRIVATE
        dexel
        CLI11::CLI11
        Eigen3::Eigen
        igl::core

)
