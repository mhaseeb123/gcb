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

/* constants and preprocessors used throughout the repo */

#pragma once

// size preprocessors

// kbytes
#define KBYTES(x)                          (1024 * (x))

// mbytes
#define MBYTES(x)                          (1024 * KBYTES(x))

// gbytes
#define GBYTES(x)                          (1024 * MBYTES(x))

// OpenMPI's GPUDirectRDMA max message size is 30,000 bytes
constexpr int OMPI_MAX_GDR_MSG_SIZE        = 30e3;

// max CUDA streams
constexpr int MAX_STREAMS                  = 4;

// max CUDA kernel events + h2d & d2h events in driver
constexpr int MAX_EVENTS                   = 2;

// max iterations
constexpr int MAX_ITERS                    = 1000;

// status success
constexpr int SUCCESS                      = 0;

// status failure (if needed)
constexpr int FAILURE                      = -1;
