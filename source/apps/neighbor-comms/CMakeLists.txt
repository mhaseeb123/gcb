project(neighbor-comms LANGUAGES CUDA)

add_executable(neighbor-comms ${_EXCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/neighbor-comms.cu)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/neighbor-comms.cu PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE CXX)

# add dependencies (not applicable yet)
# add_dependencies(apitest magic_enum argparse)

# include core/include and generated files
target_include_directories(neighbor-comms PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include ${CMAKE_BINARY_DIR}
                            ${ARGPARSE_INCLUDE_DIR})

target_link_libraries(neighbor-comms PUBLIC ${MPI_LIBS})

set_target_properties(neighbor-comms
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
install(TARGETS neighbor-comms DESTINATION ${CMAKE_INSTALL_BINDIR})