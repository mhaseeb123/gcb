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
#include "cuda.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

namespace gcb
{
namespace cuda
{

#define error_check(ans)                   check((ans), __FILE__, __LINE__)

// ------------------------------------------------------------------------------------ //

// error checking function
template <typename T>
static inline void check(T result, const char *const file, const int line, bool is_fatal = true)
{
    if (result != cudaSuccess)
    {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(result) << std::endl;

        if (is_fatal)
            exit(result);
    }
}

// ------------------------------------------------------------------------------------ //

struct gpu_manager
{
    int _ngpus = 0;

    gpu_manager()
    {
        error_check(cudaGetDeviceCount(&_ngpus));
        std::cout << "GPU Manager: Available CUDA Devices: " << _ngpus << std::endl;
    }

    int get_gpu_id()
    {
        int gpu_id;
        error_check(cudaGetDevice(&gpu_id));
        return gpu_id;
    }

    int get_ngpus()
    {
        return _ngpus;
    }

    void set_gpu_id(int id = 0)
    {
        int gpu_id = (id > 0 && id < _ngpus) ? id : 0;
        error_check(cudaSetDevice(gpu_id));
        return;
    }

    static gpu_manager& get_instance()
    {
        static gpu_manager instance;
        return instance;
    }
};

// ------------------------------------------------------------------------------------ //

// class driver
class driver
{
public:

    cudaStream_t stream[MAX_STREAMS];
    cudaEvent_t d2h;
    cudaEvent_t h2d;
    cudaEvent_t events[MAX_EVENTS];

    // ------------------------------------------------------------------------------------ //

    driver(int _gpu_id)
    {
        auto manager = gpu_manager::get_instance();

        int gpu_id = manager.get_gpu_id();

        if (_gpu_id > 0 && _gpu_id < manager.get_ngpus())
            gpu_id = _gpu_id;

        error_check(cudaSetDevice(gpu_id));
        std::cout << "DRIVER: Setting Device to: " << gpu_id << std::endl;

        for (int i = 0; i < MAX_STREAMS; i++)
            error_check(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));

        error_check(cudaEventCreateWithFlags(&d2h, cudaEventBlockingSync));

        for (int i = 0; i < MAX_EVENTS; i++)
            error_check(cudaEventCreateWithFlags(&events[i], cudaEventBlockingSync));
    }

    // ------------------------------------------------------------------------------------ //
    driver()
    {
        auto manager = gpu_manager::get_instance();
        error_check(cudaSetDevice(manager.get_gpu_id()));
        std::cout << "DRIVER: Setting Device to: " << manager.get_gpu_id() << std::endl;

        for (int i = 0; i < MAX_STREAMS; i++)
            error_check(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking));

        error_check(cudaEventCreateWithFlags(&d2h, cudaEventBlockingSync));

        for (int i = 0; i < MAX_EVENTS; i++)
            error_check(cudaEventCreateWithFlags(&events[i], cudaEventBlockingSync));
    }

    // ------------------------------------------------------------------------------------ //
    ~driver()
    {
        for (int i = 0; i < MAX_STREAMS; i++)
            error_check(cudaStreamDestroy(stream[i]));

        error_check(cudaEventDestroy(d2h));

        for (int i = 0; i < MAX_EVENTS; i++)
            error_check(cudaEventDestroy(events[i]));
    }

    // ------------------------------------------------------------------------------------ //

    void stream_sync(int i = 0)
    {
        error_check(cudaStreamSynchronize(stream[i]));
    }

    // ------------------------------------------------------------------------------------ //

    void all_streams_sync()
    {
        for (int i = 0; i < MAX_STREAMS; i++)
            error_check(cudaStreamSynchronize(stream[i]));
    }

    // ------------------------------------------------------------------------------------ //

    void event_sync(cudaEvent_t &event)
    {
        error_check(cudaEventSynchronize(event));
    }

    // ------------------------------------------------------------------------------------ //

    bool event_query(cudaEvent_t &event)
    {
        auto status = cudaEventQuery(event);

        if (status == cudaSuccess)
            return true;
        else if (status == cudaErrorNotReady)
            return false;
        else
            error_check(status);

        return false;
    }

    // ------------------------------------------------------------------------------------ //

    auto& get_stream(int i = 0) const
    {
        return stream[i];
    }

    // ------------------------------------------------------------------------------------ //

    // We can have multiple concurrent resident kernels
    // per device depending on device compute capability
    static driver* get_instance(int _gpu_id = 0)
    {
        static driver instance(_gpu_id);
        return &instance;
    }
};

// ------------------------------------------------------------------------------------ //

// CUDA driver functions
template <typename T>
auto H2D(T *&dst, T *&src, const size_t size, const cudaStream_t &stream)
{
    return cudaMemcpyAsync(dst, src, size * sizeof(T), cudaMemcpyHostToDevice, stream);
}

// ------------------------------------------------------------------------------------ //

template <typename T>
auto H2D_withEvent(T *&dst,  T *&src, const size_t size, const cudaStream_t &stream)
{
    auto drv = driver::get_instance();
    cudaMemcpyAsync(dst, src, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    return cudaEventRecord(drv->h2d, stream);
}

// ------------------------------------------------------------------------------------ //

template <typename T>
auto D2H(T *&dst, T *&src, const size_t size, const cudaStream_t &stream)
{
    return cudaMemcpyAsync(dst, src, size * sizeof(T), cudaMemcpyDeviceToHost, stream);
}

// ------------------------------------------------------------------------------------ //

template <typename T>
auto D2H_withEvent(T *&dst, T *&src, const size_t size, const  cudaStream_t &stream)
{
    auto drv = driver::get_instance();
    cudaMemcpyAsync(dst, src, size * sizeof(T), cudaMemcpyDeviceToHost, stream);
    return cudaEventRecord(drv->d2h, stream);
}

// ------------------------------------------------------------------------------------ //

template <typename T>
auto D2D(T *&dst, T *&src, const size_t size, const cudaStream_t &stream)
{
    return cudaMemcpyAsync(dst, src, size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
}

// ------------------------------------------------------------------------------------ //

template <typename T>
auto host_pinned_allocate(T *&ptr, size_t size)
{
    return cudaMallocHost(&ptr, size * sizeof(T));
}

// ------------------------------------------------------------------------------------ //

template <typename T>
auto device_allocate(T *&ptr, size_t size)
{
    return cudaMalloc(&ptr, size * sizeof(T));
}

// ------------------------------------------------------------------------------------ //

template <typename T>
auto device_allocate_async(T *&ptr, size_t size, const cudaStream_t &stream)
{
    return cudaMallocAsync(&ptr, size * sizeof(T), stream);
}

// ------------------------------------------------------------------------------------ //

template <typename T>
auto host_pinned_free(T *&ptr)
{
    return cudaFreeHost(ptr);
    ptr = nullptr;
}

// ------------------------------------------------------------------------------------ //

template <typename T>
auto device_free(T *&ptr)
{
    return cudaFree(ptr);
    ptr = nullptr;
}

// ------------------------------------------------------------------------------------ //

template <typename T>
auto device_free_async(T *&ptr, const cudaStream_t &stream)
{
    return cudaFreeAsync(ptr, stream);
}

auto device_sync()
{
    return cudaDeviceSynchronize();
}

}; //namespace cuda

}; // namespace gcb