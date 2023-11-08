project(part-comm-demo LANGUAGES CXX)

add_executable(part-comm-demo ${_EXCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/part-comm-demo.cpp)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/part-comm-demo.cpp PROPERTIES LANGUAGE CXX LINKER_LANGUAGE CXX)

# include core/include and generated files

target_include_directories(part-comm-demo PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include ${CMAKE_BINARY_DIR}
                            ${ARGPARSE_INCLUDE_DIR})

target_link_libraries(part-comm-demo PUBLIC ${MPI_LIBS})

set_target_properties(part-comm-demo
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
install(TARGETS part-comm-demo DESTINATION ${CMAKE_INSTALL_BINDIR})