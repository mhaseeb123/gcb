
#----------------------------------------------------------------------------------------#
#   Add apitest benchmark
#----------------------------------------------------------------------------------------#

message(STATUS "Adding apitest...")
add_subdirectory(apitest)

#----------------------------------------------------------------------------------------#
#   Add GPU-GPU bandwidth bechmark
#----------------------------------------------------------------------------------------#

message(STATUS "Adding GPU-GPU bandwidth bechmark...")
add_subdirectory(bandwidth)

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