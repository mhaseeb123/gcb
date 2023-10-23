project(mpi-cuda-hello LANGUAGES CXX)

add_executable(mpi-cuda-hello ${_EXCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/mpi-cuda-hello.cpp)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/mpi-cuda-hello.cpp PROPERTIES LANGUAGE CXX LINKER_LANGUAGE CXX)

# include core/include and generated files
target_include_directories(mpi-cuda-hello PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include ${CMAKE_BINARY_DIR}
                            ${ARGPARSE_INCLUDE_DIR})

target_link_libraries(mpi-cuda-hello PUBLIC ${MPI_LIBS} cudart)

set_target_properties(mpi-cuda-hello
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
install(TARGETS mpi-cuda-hello DESTINATION ${CMAKE_INSTALL_BINDIR})