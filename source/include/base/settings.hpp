#pragma once

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


// settings data structure
struct settings
{
    size_t msg_size;
    size_t freq;
    size_t nthreads;
    size_t niters;

    settings()
    {
        msg_size = freq = nthreads = niters = 0;
    }

    ~settings() = default;

    // copy constructor
    settings(size_t _msg_size, size_t _freq, size_t _nthreads, size_t _niters)
        : msg_size(_msg_size), freq(_freq), nthreads(_nthreads), niters(_niters) {};

    // overload = operator
    settings& operator=(settings &&_settings)
    {
        this->msg_size = _settings.msg_size;
        this->freq = _settings.freq;
        this->nthreads = _settings.nthreads;
        this->niters = _settings.niters;
        return *this;
    }
};

using knobs_t = knobs;
using settings_t = settings;