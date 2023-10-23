project(apitest LANGUAGES CXX)

add_executable(apitest ${_EXCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/apitest.cpp)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/apitest.cpp PROPERTIES LANGUAGE CXX LINKER_LANGUAGE CXX)

# add dependencies (not applicable yet)
# add_dependencies(apitest magic_enum argparse)

# include core/include and generated files
# DO NOT include ${MAGICENUM_INCLUDE_DIR} as it results in an internal error at templates due to CUDA compiler.
target_include_directories(apitest PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include ${CMAKE_BINARY_DIR}
                            ${ARGPARSE_INCLUDE_DIR})

target_link_libraries(apitest PUBLIC ${MPI_LIBS})

set_target_properties(apitest
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