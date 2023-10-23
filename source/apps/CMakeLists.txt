
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

message(STATUS "Adding mpi cuda hello demo...")
add_subdirectory(mpi-cuda-hello)

#----------------------------------------------------------------------------------------#
#   Add MPI Flood Send Put benchmark
#----------------------------------------------------------------------------------------#

message(STATUS "Adding neighbor comms benchmark...")
add_subdirectory(neighbor-comms)