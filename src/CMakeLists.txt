add_library(dexel INTERFACE)

target_include_directories(dexel
    INTERFACE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
