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

#include "gcb.hpp"
#include <execution>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <memory>

#include "commlibs/mpi/driver.hpp"
#include "cuda/driver.hpp"
//#include <helper_cuda.h>

#define checkCudaErrors(ans)                   check_err((ans), __FILE__, __LINE__)

// error checking function
template <typename T>
static inline void check_err(T result, const char *const file, const int line, bool is_fatal = true)
{
    if (result != cudaSuccess)
    {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(result) << std::endl;

        if (is_fatal)
            exit(result);
    }
}

struct CudaMallocDeleter {
  void operator()(double* obj) {
    checkCudaErrors(cudaFree(obj));
  }
};


enum NeighborIndex {
  XPLUS = 0,
  XMINUS = 1,
  YPLUS = 2,
  YMINUS = 3,
  ZPLUS = 4,
  ZMINUS = 5
};