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

/* Helper functions and algorithms */

#pragma once

#include <omp.h>
#include "gcb.hpp"
#include "cuda/driver.hpp"

using namespace gcb::cuda;

namespace gcb
{

namespace algs
{

template <typename T>
__host__ T *&get_hdata(size_t size = 0)
{
    static T *h_data = nullptr;

    if (h_data == nullptr)
    {
        if (size == 0)
            throw std::runtime_error("Cannot allocate zero size data");
        else
            h_data = error_check(host_pinned_allocate(h_data, size));
    }

    return h_data;
}

template<typename T>
__host__ void free_hdata()
{
    T &&h_data = get_hdata<T>();

    if (h_data != nullptr)
    {
        error_check(host_pinned_free(h_data));
        h_data = nullptr;
    }
}

template <typename T>
__host__ T *&get_ddata(size_t size = 0, int stream = 0)
{
    static T *d_data = nullptr;

    if (d_data == nullptr)
    {
        auto &cuda_drv = driver::get_instance();
        if (size == 0)
            throw std::runtime_error("Cannot allocate zero size data");
        else
            d_data = error_check(device_allocate_async(d_data, size, cuda_drv->get_stream(stream)));
    }

    return d_data;
}

template<typename T>
__host__ void free_ddata(int stream = 0)
{
    T &&d_data = get_ddata<T>();

    if (d_data != nullptr)
    {
        auto &cuda_drv = driver::get_instance();
        error_check(device_free_async(d_data, cuda_drv->get_stream(stream)));
        d_data = nullptr;
    }
}

__host__ void seed_random()
{
    // only seed once
    static bool done = [](){
        srand(time(nullptr));
        return true;
    } ();
}

template<typename T>
__host__ void generate_random_data(T *data, size_t size, int range = std::numeric_limits<T>::max())
{
    // generate random data
    if constexpr(std::is_arithmetic_v<T>)
    {
        // range cannot be larger than the numeric limit of T
        range = std::min(range, std::numeric_limits<T>::max());

        // check for underflow as well
        range = std::max(range, std::numeric_limits<T>::min());

        // use max threads to init data with openmp
#pragma omp parallel for num_threads(omp_get_max_threads()) \
                         schedule(static) default(none)     \
                         shared(data, size, range)
        // generate random data
        for (int i = 0; i < size; i++)
            data[i] = rand() % range;
    }
}

template <typename T>
__global__ void vecAdd(T *a, T *b, T *c, size_t n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Make sure we do not go out of bounds
    for (; id < n; id += stride)
        c[id] = a[id] + b[id];
}

} // namespace algs
} // namespace gcb