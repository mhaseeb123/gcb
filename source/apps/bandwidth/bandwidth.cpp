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

#include "bandwidth.hpp"

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

    auto *cuda_drv = gcb::cuda::driver::get_instance(rank%4);

    /* -------------------------------------------------------------------------------------------
        Loop from 8 B to 1 GB
    --------------------------------------------------------------------------------------------*/

    for(int i=0; i<=27; i++){

        long int N = 1 << i;

        // Allocate memory for A on CPU
        static double *A = new double[1 << 27];

        // Initialize all elements of A to random values
        if (i == 0)
        {
            for(int j=0; j<N; j++)
                A[j] = (double)rand()/(double)RAND_MAX;
        }

        static double *d_A = nullptr;

        if (d_A == nullptr)
        {
            gcb::cuda::error_check(gcb::cuda::device_allocate(d_A, 1 << 27));
        }

        gcb::cuda::error_check(gcb::cuda::H2D(d_A, A, N, cuda_drv->get_stream()));

        cuda_drv->stream_sync();

        int tag1 = 10;
        int tag2 = 20;

        int loop_count = 50;

        // Warm-up loop
        for(int j=1; j<=5; j++){
            if(rank == 0){
                mpi_driver->Send(d_A, N, MPI_DOUBLE, 1, tag1);
                status = mpi_driver->Recv(d_A, N, MPI_DOUBLE, 1, tag2);
            }
            else if(rank == 1){
                status = mpi_driver->Recv(d_A, N, MPI_DOUBLE, 0, tag1);
                mpi_driver->Send(d_A, N, MPI_DOUBLE, 0, tag2);
            }
        }

        // Time ping-pong for loop_count iterations of data transfer size 8*N bytes
        double start_time, stop_time, elapsed_time;
        start_time = mpi_driver->Wtime();

        for(int j=1; j<=loop_count; j++){
            if(rank == 0){
                mpi_driver->Send(d_A, N, MPI_DOUBLE, 1, tag1);
                status = mpi_driver->Recv(d_A, N, MPI_DOUBLE, 1, tag2);
            }
            else if(rank == 1){
                status = mpi_driver->Recv(d_A, N, MPI_DOUBLE, 0, tag1);
                mpi_driver->Send(d_A, N, MPI_DOUBLE, 0, tag2);
            }
        }

        stop_time = mpi_driver->Wtime();
        elapsed_time = stop_time - start_time;

        long int num_B = 8*N;
        long int B_in_GB = 1 << 30;
        double num_GB = (double)num_B / (double)B_in_GB;
        double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

        if(rank == 0) printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );

        if (i == 27)
        {
            gcb::cuda::error_check(gcb::cuda::device_free(d_A));
            delete[] A;
        }
    }

    mpi_driver->finalize();
}