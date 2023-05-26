project(apitest LANGUAGES CUDA)

add_executable(at_mpi ${_EXCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/at_mpi.cu)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/at_mpi.cu PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE CUDA)

# add dependencies (not applicable yet)
# add_dependencies(at_mpi magic_enum argparse)

# include core/include and generated files
# DO NOT include ${MAGICENUM_INCLUDE_DIR} as it results in an internal error at templates due to CUDA compiler.
target_include_directories(at_mpi PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include ${CMAKE_BINARY_DIR}
                            ${ARGPARSE_INCLUDE_DIR})

target_link_libraries(at_mpi PUBLIC ${MPI_LIBS})

set_target_properties(at_mpi
    PROPERTIES
        CUDA_ARCHITECTURES 60 70 80
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        INSTALL_RPATH_USE_LINK_PATH ON
)



# installation
install(TARGETS at_mpi DESTINATION ${CMAKE_INSTALL_BINDIR})