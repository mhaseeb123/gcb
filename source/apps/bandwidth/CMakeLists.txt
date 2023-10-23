project(bandwidth LANGUAGES CXX)

add_executable(bandwidth ${_EXCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/bandwidth.cpp)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/bandwidth.cpp PROPERTIES LANGUAGE CXX LINKER_LANGUAGE CXX)

# include core/include and generated files
target_include_directories(bandwidth PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include ${CMAKE_BINARY_DIR}
                            ${ARGPARSE_INCLUDE_DIR})

target_link_libraries(bandwidth PUBLIC ${MPI_LIBS} cudart)

set_target_properties(bandwidth
    PROPERTIES
        CXX_STANDARD ${CMAKE_CXX_STANDARD}
        CUDA_ARCHITECTURES 80
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        INSTALL_RPATH_USE_LINK_PATH ON
)

# installation
install(TARGETS apitest DESTINATION ${CMAKE_INSTALL_BINDIR})