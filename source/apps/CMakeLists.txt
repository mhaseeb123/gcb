
#----------------------------------------------------------------------------------------#
#   Add apitest benchmark
#----------------------------------------------------------------------------------------#

message(STATUS "Adding apitest...")
add_subdirectory(apitest)

#----------------------------------------------------------------------------------------#
#   Add broadcast benchmarks
#----------------------------------------------------------------------------------------#

message(STATUS "Adding mpi partitioned communication demo...")
add_subdirectory(mpi-part-comm-demo)