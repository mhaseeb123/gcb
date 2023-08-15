project(flood-send-put-num LANGUAGES CUDA)

add_executable(flood-send-put-num ${_EXCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/flood_send_put_num.cpp)

set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/flood_send_put_num.cpp PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE CUDA)

# add dependencies (not applicable yet)
# add_dependencies(flood-send-put-num magic_enum argparse)

# include core/include and generated files
# DO NOT include ${MAGICENUM_INCLUDE_DIR} as it results in an internal error at templates due to CUDA compiler.
target_include_directories(flood-send-put-num PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include ${CMAKE_BINARY_DIR}
                            ${ARGPARSE_INCLUDE_DIR})

target_link_libraries(flood-send-put-num PUBLIC ${MPI_LIBS})

set_target_properties(flood-send-put-num
    PROPERTIES
        CXX_STANDARD ${CXX_STANDARD}
        CXX_EXTENSIONS NO
        INSTALL_RPATH_USE_LINK_PATH ON
)

# installation
install(TARGETS flood-send-put-num DESTINATION ${CMAKE_INSTALL_BINDIR})