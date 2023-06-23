
#----------------------------------------------------------------------------------------#
#   Add apitest benchmark
#----------------------------------------------------------------------------------------#

message(STATUS "Adding apitest...")
add_subdirectory(apitest)

#----------------------------------------------------------------------------------------#
#   Add MPI partial communication demo
#----------------------------------------------------------------------------------------#

message(STATUS "Adding mpi partitioned communication demo...")
add_subdirectory(mpi-part-comm-demo)

#----------------------------------------------------------------------------------------#
#   Add MPI Flood Send Put benchmark
#----------------------------------------------------------------------------------------#

message(STATUS "Adding mpi flood send put benchmark...")
add_subdirectory(flood-send-put-num)