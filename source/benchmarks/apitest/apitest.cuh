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

#pragma once

#include "cuda/driver.hpp"

namespace gcb
{
namespace alg
{
namespace at
{

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

// if constexpr(std::is_arithmetic_v<T>)

template<typename T>
auto test_vecAdd(int rank, int size, const bool AWARE)
{
    status_t status = 0;

        // Host input vectors
    T *h_a;
    T *h_b;
    //Host output vector
    T *h_c;

    // Device input vectors
    T *d_a;
    T *d_b;
    //Device output vector
    T *d_c;

    int n = 1000;
    int bytes = n * sizeof(T);

   // Allocate memory for each vector on host
    h_a = (T*)malloc(bytes);
    h_b = (T*)malloc(bytes);
    h_c = (T*)malloc(bytes);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = sin(i)*sin(i);
        h_b[i] = cos(i)*cos(i);
    }

    // Copy host vectors to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n/blockSize);

    if (rank == 0)
        // Execute the kernel
        vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    cudaDeviceSynchronize();

    // send a message
    if (rank == 0)
    {
        if (!AWARE)
        {
            // Copy array back to host
            cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
            status = MPI_Send(h_c, n, MPI_DOUBLE, 1, 0x99, MPI_COMM_WORLD);
        }
        else
        {
            status = MPI_Send(d_c, n, MPI_DOUBLE, 1, 0x99, MPI_COMM_WORLD);
        }
    }
    // receive a message
    else
    {
        if (!AWARE)
        {
            status = MPI_Recv(h_c, n, MPI_DOUBLE, 0, 0x99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
        {
            status = MPI_Recv(d_c, n, MPI_DOUBLE, 0, 0x99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Copy array back to host
            cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
        }

        // Sum up vector c and print result divided by n, this should equal 1 within error
        double sum = 0;
        for(i=0; i<n; i++)
            sum += h_c[i];

        printf("final result: %f\n", sum/n);
    }

    return status;
}

#if 0
template<typename T, std::enable_if<std::is_arithmetic<T>::value, T>::type>
auto test_p2p(knobs &k, bool async = false)
{
    // compute the range
    double t_min = 1.0/k.fmin;
    double t_max = 1.0/k.fmax;
    double t_step = 1.0/k.fstep;

    // allocate pinned memory for msg_max
    T *raw_data = nullptr;
    gcb::cuda::host_pinned_allocate(raw_data, k.msg_max);

    // make unique pointer to raw data
    std::unique_ptr<T> data = make_unique<T>(raw_data);

    // generate random data
    gcb::alg::generate_random_data(data.get(), k.msg_max);

}
#endif

} // namespace at
} // namespace alg
} // namespace gcb


