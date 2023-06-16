project(pcommdemo LANGUAGES CUDA)

add_executable(pcommdemo ${_EXCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/pcommdemo.cu)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/pcommdemo.cu PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE CUDA)

# add dependencies (not applicable yet)
# add_dependencies(pcommdemo magic_enum argparse)

# include core/include and generated files
# DO NOT include ${MAGICENUM_INCLUDE_DIR} as it results in an internal error at templates due to CUDA compiler.
target_include_directories(pcommdemo PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include ${CMAKE_BINARY_DIR}
                            ${ARGPARSE_INCLUDE_DIR})

target_link_libraries(pcommdemo PUBLIC ${MPI_LIBS})

set_target_properties(pcommdemo
    PROPERTIES
        CXX_STANDARD ${CXX_STANDARD}
        CUDA_ARCHITECTURES 60 70 80
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        INSTALL_RPATH_USE_LINK_PATH ON
)



# installation
install(TARGETS pcommdemo DESTINATION ${CMAKE_INSTALL_BINDIR})