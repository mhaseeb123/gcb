##########################################################################################
#       CMake settings
##########################################################################################

# cmake min required
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(gcb)

# in source build warning
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintented, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(WARNING "In-source build")
endif()

# set cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/external/")

# set these as the defaults
set(CMAKE_ENABLE_EXPORTS ON CACHE BOOL "Executable exports symbols for loadable modules")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON CACHE BOOL "Append directories in the linker search path")

# Set a default build type if none was specified
set(GCB_BUILD_TYPE "RelWithDebInfo")

# set the build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${GCB_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${GCB_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

##########################################################################################
#       CMake settings
##########################################################################################
option(USE_NCCL "Enable NCCL usage" OFF)
option(USE_NVSHMEM "Enable NVSHMEM usage" OFF)
option(USE_TIMEMORY "Enable Timemory instrumentation" OFF)
option(USE_MPIP_LIBRARY "Enable MPIP instrumentation via Timemory" OFF)

##########################################################################################
#       GCC version check
##########################################################################################
set(GCC_EXPECTED_VERSION 11.7)

if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS GCC_EXPECTED_VERSION)
    message(FATAL_ERROR "GCC: GCB requires GCC v${GCC_EXPECTED_VERSION} or higher to build but found v${CMAKE_CXX_COMPILER_VERSION}")
endif()

##########################################################################################
#       CXX standard
##########################################################################################
set(CXX_STANDARD_REQUIRED ON)

# required minimum CXX standard
set(CMAKE_CXX_STANDARD_REQUIRED 20)

if(NOT CXX_STANDARD OR (CXX_STANDARD LESS ${CMAKE_CXX_STANDARD_REQUIRED}))
    set(CXX_STANDARD ${CMAKE_CXX_STANDARD_REQUIRED})
    message(STATUS "Setting CXX_STANDARD to ${CMAKE_CXX_STANDARD_REQUIRED}")
endif()

##########################################################################################
#       NCCL and NVSHMEM
##########################################################################################
if (USE_NCCL)
    find_package(NCCL REQUIRED)
endif()

if(NCCL_FOUND AND USE_NCCL)
    list(APPEND _NCCL NCCL::NCCL_CXX)
    include_directories(SYSTEM ${NCCL_INCLUDE_PATH})
    target_compile_definitions(NCCL::NCCL_CXX INTERFACE)
    set(TMP ${TMP} nccl)
endif()

if (USE_NVSHMEM)
    find_package(NVSHMEM REQUIRED)
endif()

if(NVSHMEM_FOUND AND USE_NVSHMEM)
    list(APPEND _NVSHMEM NVSHMEM::NVSHMEM_CXX)
    include_directories(SYSTEM ${NVSHMEM_INCLUDE_PATH})
    target_compile_definitions(NVSHMEM::NVSHMEM_CXX INTERFACE)
    set(TMP ${TMP} nvshmem)
endif()

##########################################################################################
#       Configure Instrumentation
##########################################################################################
if (USE_TIMEMORY)
    set(timemory_FIND_COMPONENTS_INTERFACE timemory-gcb)
    set(COMPONENTS headers compile-options analysis-tools ${TMP}

        OPTIONAL_COMPONENTS cxx)

    # Find Timemory package
    find_package(timemory COMPONENTS ${COMPONENTS} OPTIONAL_COMPONENTS ${OPTIONAL_COMPONENTS})

    # if timemory package found then find mpip library
    if (timemory_DIR)
        file(MAKE_DIRECTORY $ENV{HOME}/mplconfigdir)
        message(STATUS "Timemory component interface enabled")
    else()
        message(WARNING "Timemory not found => Setting: USE_TIMEMORY=OFF")
        set(USE_TIMEMORY OFF CACHE BOOL "Timemory not found, disabling instrumentation" FORCE)
    endif()
else()
    message(STATUS "Timemory component interface disabled")
endif()

# safety check for MPIP library
if (USE_MPIP_LIBRARY AND NOT (USE_TIMEMORY AND GCB_USE_MPI))
    message(WARNING "MPIP instrumentation disabled. USE_MPIP_LIBRARY=ON requires USE_TIMEMORY=ON and USE_MPI=ON")
    set(USE_MPIP_LIBRARY OFF CACHE BOOL "Disabling MPIP instrumentation." FORCE)
endif()

# Find timemory-mpip.so library if required
if (USE_MPIP_LIBRARY)
    find_library(MPIP_LIBRARY
                 NAMES timemory-mpip
                 HINTS ENV PATH
                       ENV LD_LIBRARY_PATH
                       ENV CMAKE_PREFIX_PATH
                 PATH_SUFFIXES lib lib64
                 DOC "Timemory MPIP instrumentation library"
                )
    if (MPIP_LIBRARY)
        message(STATUS "Timemory-MPIP library interface enabled")
    else ()
        message(WARNING "Timemory-MPIP library not found => Setting: USE_MPIP_LIBRARY=OFF")
        set(USE_MPIP_LIBRARY OFF CACHE BOOL "Timemory-MPIP library not found" FORCE)
    endif()
else()
    message(STATUS "Timemory-MPIP library interface disabled")
endif()

##########################################################################################
#       Check and set CMAKE_CUDA_HOST_COMPILER
##########################################################################################

if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    message(WARNING "CMAKE_CUDA_HOST_COMPILER not provided\n=> Setting CMAKE_CUDA_HOST_COMPILER=$(which CC)")

    # run which CC to find the CUDA_HOST_COMPILER
    execute_process(
        COMMAND bash -c "which CC"
        OUTPUT_VARIABLE CUDA_HOST_COMPILER)

    # strip the last \n from the string
    string(STRIP ${CUDA_HOST_COMPILER} CUDA_HOST_COMPILER)

    # check for empty string
    if (NOT CUDA_HOST_COMPILER STREQUAL "")
        # set the CMAKE_CUDA_HOST_COMPILER and cache it for future use
        set(CMAKE_CUDA_HOST_COMPILER CACHE STRING ${CUDA_HOST_COMPILER})
        message(STATUS "Setting CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}")
    else()
        message(FATAL_ERROR "CMAKE_CUDA_HOST_COMPILER not found. Please set it manually to CC or provide via CMAKE_CUDA_HOST_COMPILER")
    endif()

endif()

##########################################################################################
#       Configure Header files
##########################################################################################
message(STATUS "Configuring...")

# configure the file: config.hpp
configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/config.hpp.in config.hpp @ONLY)

# install the configured file as well
install(FILES ${CMAKE_BINARY_DIR}/aliases.hpp ${CMAKE_BINARY_DIR}/config.hpp DESTINATION include)


##########################################################################################
#       Add sub-directories
##########################################################################################

#----------------------------------------------------------------------------------------#
#   externals
#----------------------------------------------------------------------------------------#

#message(STATUS "Adding external...")
#add_subdirectory(external)

#----------------------------------------------------------------------------------------#
#   hicops source
#----------------------------------------------------------------------------------------#

message(STATUS "Adding gcb source...")
add_subdirectory(source)