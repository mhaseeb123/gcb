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

/* common stdlib includes */

#pragma once

#include <iostream>
#include <numeric>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <string>
#include <thread>
#include <semaphore.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>
#include <functional>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <queue>
#include <stack>
#include <list>
#include <memory>
#include "constants.hpp"
#include <math.h>

// type aliases
using status_t = int;


// knobs data structure
struct knobs
{
    knobs()
    {
        fmin = fmax = fstep = msg_min = msg_max = \
        msg_step = niters = nthreads = 1;
    };

    ~knobs() = default;

    knobs(size_t _fmin, size_t _fmax, size_t _fstep, size_t _msg_min,
        size_t _msg_max, size_t _msg_step, size_t _niters, size_t _nthreads)
        : fmin(_fmin), fmax(_fmax), fstep(_fstep), msg_min(_msg_min),
        msg_max(_msg_max), msg_step(_msg_step), niters(_niters), nthreads(_nthreads) {};

    // copy constructor
    knobs(knobs &_knobs)
    {
        this->fmin = _knobs.fmin;
        this->fmax = _knobs.fmax;
        this->fstep = _knobs.fstep;
        this->msg_min = _knobs.msg_min;
        this->msg_max = _knobs.msg_max;
        this->msg_step = _knobs.msg_step;
        this->niters = _knobs.niters;
        this->nthreads = _knobs.nthreads;
    }

    // overload = operator
    knobs& operator=(knobs &&_knobs)
    {
        this->fmin = _knobs.fmin;
        this->fmax = _knobs.fmax;
        this->fstep = _knobs.fstep;
        this->msg_min = _knobs.msg_min;
        this->msg_max = _knobs.msg_max;
        this->msg_step = _knobs.msg_step;
        this->niters = _knobs.niters;
        this->nthreads = _knobs.nthreads;
        return *this;
    }

    void sanitize()
    {
        // basic checks

        // min and max freqs
        if (fmin > fmax)
            std::swap(fmin, fmax);

        // min and max message sizes
        if (msg_min > msg_max)
            std::swap(msg_min, msg_max);

        // frequency stepping
        if (fstep == 0)
            fstep = 1;
        else if (fstep > fmax - fmin)
            fstep = fmax - fmin;

        // message size stepping
        if (msg_step == 0)
            msg_step = 1;
        else if (msg_step > msg_max - msg_min)
            msg_step = msg_max - msg_min;

        // minimum 1 thread
        if (nthreads < 1)
            nthreads = 1;

        // minimum 1 iteration
        if (niters < 1)
            niters = 1;
        else if (niters > MAX_ITERS)
            niters = MAX_ITERS;
    }

    size_t fmin;
    size_t fmax;
    size_t fstep;
    size_t msg_min;
    size_t msg_max;
    size_t msg_step;
    size_t niters;
    size_t nthreads;
};