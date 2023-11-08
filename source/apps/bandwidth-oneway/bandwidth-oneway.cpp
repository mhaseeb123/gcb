/*
 * MIT License
 *
 * Copyright (c) 2023 The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of any
 * required approvals from the U.S. Dept. of Energy).  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// This code has been modified from: https://github.com/olcf-tutorials/MPI_ping_pong
// Please see the original licensing information at the OLCF GitHub repository
// for more information

#include "bandwidth-oneway.hpp"

// number of times each transfer should run
constexpr int loop_count = 50;

status_t main(int argc, char* argv[])
{
    /* -------------------------------------------------------------------------------------------
        MPI Initialization
    --------------------------------------------------------------------------------------------*/
    auto *mpi_driver = gcb::comm::mpi::driver::get_instance();

    // initialize MPI
    mpi_driver->init(argc, argv);

    // get rank and size
    int rank = mpi_driver->rank();
    int size = mpi_driver->size();

    MPI_Status status;

    if(size != 2)
    {
        if(rank == 0)
            printf("This program requires exactly 2 MPI ranks, but you are attempting to use %d! Exiting...\n", size);

        mpi_driver->finalize();
        exit(0);
    }

    /* -------------------------------------------------------------------------------------------
        CUDA Initialization
    --------------------------------------------------------------------------------------------*/
    auto *cuda_drv = gcb::cuda::driver::get_instance(rank%4);

    /* -------------------------------------------------------------------------------------------
        Experimental Setup
    --------------------------------------------------------------------------------------------*/
    // bytes calculations

    // 1GB in bytes
    constexpr long int B_in_GB = 1 << 30;

    // niterations = log2(1GB) - log2(sizeof(double))
    constexpr long int niters = static_cast<int>(log2(B_in_GB)) - static_cast<int>(log2(sizeof(double)));

    // create host and device arrays
    double *h_Arr = nullptr;
    double *d_Arr = nullptr;

    // Allocate memory on CPU and GPU
    gcb::cuda::error_check(gcb::cuda::host_pinned_allocate(h_Arr, B_in_GB));
    gcb::cuda::error_check(gcb::cuda::device_allocate(d_Arr, B_in_GB));

    // generate random vector in h_Arr
    gcb::bw_oway::genRandomVector<double>(h_Arr, B_in_GB, 0.0, 1e7);

    /* -------------------------------------------------------------------------------------------
        MPI_Put: Loop from 8B to 1GB
    --------------------------------------------------------------------------------------------*/
    // print status
    if (!rank)
        std::cout << "\n\nRunning MPI_Put bandwidth test:\n\n" << std::flush;

    for(int i=0; i<=niters; i++)
    {
        long int N = 1 << i;

        if (!rank)
        {
            gcb::cuda::error_check(gcb::cuda::H2D(d_Arr, h_Arr, N, cuda_drv->get_stream()));
            cuda_drv->stream_sync();
        }

        // create MPI windows for one-way communication
        MPI_Win window;
        MPI_Win_create(d_Arr, N * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &window);

        // make sure the windows have been created
        MPI_Win_fence(0, window);

        // Warm-up loop
        for(int j=1; j<=5; j++)
        {
            if(!rank)
                MPI_Put(d_Arr, N, MPI_DOUBLE, 1, 0, N, MPI_DOUBLE, window);
            MPI_Win_fence(0, window);
        }

        // Time MPI_Put for loop_count iterations of data transfer size sizeof(double)*N bytes
        double start_time, stop_time, elapsed_time;
        start_time = mpi_driver->Wtime();

        // Warm-up loop
        for(int j=1; j<=loop_count; j++)
        {
            if(rank == 0)
                MPI_Put(d_Arr, N, MPI_DOUBLE, 1, 0, N, MPI_DOUBLE, window);
            MPI_Win_fence(0, window);
        }

        stop_time = mpi_driver->Wtime();
        elapsed_time = stop_time - start_time;

        long int num_B = sizeof(double)*N;
        double num_GB = (double)num_B / (double)B_in_GB;
        double avg_time_per_transfer = elapsed_time / ((double)loop_count);

        if(!rank)
            printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );

        // free window
        MPI_Win_free(&window);
    }

    /* -------------------------------------------------------------------------------------------
        MPI_Get: Loop from 8B to 1GB
    --------------------------------------------------------------------------------------------*/
    // print status
    if (!rank)
        std::cout << "\n\nRunning MPI_Get bandwidth test:\n\n" << std::flush;

    for(int i=0; i<=niters; i++)
    {
        long int N = 1 << i;

        if (rank)
        {
            gcb::cuda::error_check(gcb::cuda::H2D(d_Arr, h_Arr, N, cuda_drv->get_stream()));
            cuda_drv->stream_sync();
        }

        // create MPI windows for one-way communication
        MPI_Win window;
        MPI_Win_create(d_Arr, N * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &window);

        // make sure the windows have been created
        MPI_Win_fence(0, window);

        // Warm-up loop
        for(int j=1; j<=5; j++)
        {
            if(!rank)
                MPI_Put(d_Arr, N, MPI_DOUBLE, 1, 0, N, MPI_DOUBLE, window);
            MPI_Win_fence(0, window);
        }

        // Time MPI_Get for loop_count iterations of data transfer size sizeof(double)*N bytes
        double start_time, stop_time, elapsed_time;
        start_time = mpi_driver->Wtime();

        // Warm-up loop
        for(int j=1; j<=loop_count; j++)
        {
            if(rank == 0)
                MPI_Get(d_Arr, N, MPI_DOUBLE, 1, 0, N, MPI_DOUBLE, window);
            MPI_Win_fence(0, window);
        }

        stop_time = mpi_driver->Wtime();
        elapsed_time = stop_time - start_time;

        long int num_B = sizeof(double)*N;
        double num_GB = (double)num_B / (double)B_in_GB;
        double avg_time_per_transfer = elapsed_time / ((double)loop_count);

        if(!rank)
            printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer);

        // free window
        MPI_Win_free(&window);
    }

    /* -------------------------------------------------------------------------------------------
        Finalize
    --------------------------------------------------------------------------------------------*/
    // free memories
    gcb::cuda::error_check(gcb::cuda::device_free(d_Arr));
    gcb::cuda::error_check(gcb::cuda::host_pinned_free(h_Arr));

    mpi_driver->finalize();
}