add_library(dexel INTERFACE)

target_include_directories(dexel
    INTERFACE
        ${CMAKE_SOURCE_DIR}/include
)

# （必要があればライブラリもここでリンク）
target_link_libraries(dexel
    INTERFACE
        # CLI11::CLI11
        Eigen3::Eigen
        igl::core
)
