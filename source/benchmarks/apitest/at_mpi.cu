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

/* MPI API warmup */
#include "mpi.h"
#include "apitest.cuh"

// control if MPI is aware of CUDA
const bool AWARE = true;

void vecAdd_wrap(double *a, double *b, double *c, int n);

int main(int argc, char* argv[])
{
    status_t status = 0;

    int rank = 0;
    int size = 0;

    // initialize MPI
    status = MPI_Init(&argc, &argv);
    status = MPI_Comm_size(MPI_COMM_WORLD, &size);
    status = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //status = test_p2p(rank, size);
    gcb::alg::at::test_vecAdd<double>(rank, size, AWARE);

    status = MPI_Barrier(MPI_COMM_WORLD);
    status = MPI_Finalize();

    return status;
}