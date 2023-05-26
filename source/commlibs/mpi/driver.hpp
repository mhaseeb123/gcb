#pragma once

// include common stdlib headers
#include "commons.hpp"

// include mpi
#include <mpi.h>
#include "cuda/driver.hpp"

namespace gcb
{

// MPI driver class
class mpi : public driver<mpi>
{

public:
    mpi()
    {
        _rank = -1;
        _size = 1;
        _comm = MPI_COMM_WORLD;
    };

    ~mpi() = default;

    // initialize MPI
    MPI_Status init(int &argc, char **argv)
    {
        return MPI_Init(std::forward<int>(argc), std::forward<char**>(argv));
    }

    MPI_Status finalize()
    {
        return MPI_Finalize();
    }

    MPI_Status sync(MPI_Comm comm = _comm)
    {
        return MPI_Barrier(comm);
    }

    int rank()
    {
        return _rank;
    }

    size_t size()
    {
        return _size;
    }

    MPI_Comm& comm()
    {
        return _comm;
    }

    template <typename T>
    MPI_Status send(const T *data, size_t size = 1, MPI_Datatype type, int dst, int tag, MPI_Comm comm = _comm)
    {
        // make unique ptr for auto cleanup
        std::unique_ptr<T> h_data = make_unique<T>(new T[size]);
        // copy data to host
        cuda::H2D(h_data, data, size);
        return MPI_Send(static_cast<const void *>(h_data), size, type, dst, tag, comm);
    }

private:

    // privates
    int _rank;
    size_t _size;
    MPI_Comm _comm;
};
