
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
#   Add GPU-GPU bandwidth bechmark
#----------------------------------------------------------------------------------------#

message(STATUS "Adding one-way GPU-GPU bandwidth bechmark...")
add_subdirectory(bandwidth-oneway)

#----------------------------------------------------------------------------------------#
#   Add MPI CUDA hello demo
#----------------------------------------------------------------------------------------#

message(STATUS "Adding mpi cuda hello demo...")
add_subdirectory(mpi-cuda-hello)

#----------------------------------------------------------------------------------------#
#   Add GPU neighbor comms benchmark
#----------------------------------------------------------------------------------------#

message(STATUS "Adding neighbor comms benchmark...")
add_subdirectory(neighbor-comms)

#----------------------------------------------------------------------------------------#
#   Add MPI partitioned communications benchmark
#----------------------------------------------------------------------------------------#

message(STATUS "Adding partitioned communications benchmark...")
add_subdirectory(part-comm-demo)