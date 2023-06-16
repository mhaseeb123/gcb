#pragma once

// include common stdlib headers
#include "gcb.hpp"

// include mpi
#include <mpi.h>
#include "cuda/driver.hpp"

namespace gcb
{
namespace comm
{
namespace mpi
{

// MPI driver class
class driver
{

public:
    driver()
    {
        _rank = -1;
        _size = 1;
        _comm = MPI_COMM_WORLD;
        _aware = true;
    };

    driver(int &argc, char **argv, MPI_Comm comm  = MPI_COMM_WORLD, bool aware = true, int reqd = MPI_THREAD_MULTIPLE)
    {
        _rank = -1;
        _size = 1;
        _comm = comm;
        _aware = aware;

        this->init(argc, argv, reqd);
    };

    ~driver() = default;

    // initialize MPI
    status_t init(int argc, char **argv, int reqd = MPI_THREAD_MULTIPLE)
    {
        status_t status;

        // call init only once
        static bool done = [&]() {
            int provided = 0;
            status = MPI_Init_thread(&argc, &argv, reqd, &provided);

            if (provided != reqd)
            {
                std::cout << "FATAL: MPI_Init_thread() did not provide the requested thread support" << std::endl;
                return  FAILURE;
            }
            MPI_Comm_rank(_comm, &_rank);
            MPI_Comm_size(_comm, &_size);

            return SUCCESS;
        }();

        return done;
    }

    status_t finalize()
    {
        _rank = -1;
        _size = 0;
        return static_cast<status_t>(MPI_Finalize());
    }

    auto barrier() { return MPI_Barrier(this->_comm); }

    auto rank() { return _rank; }

    auto size() { return _size; }

    MPI_Comm& comm() { return _comm; }

    template <typename T>
    auto Send(const T *data, size_t size, MPI_Datatype type, int dst, int tag)
    {
        return MPI_Send(static_cast<const void *>(data), size, type, dst, tag, this->_comm);
    }

    template <typename T>
    auto Recv(T *data, size_t size, MPI_Datatype type, int src, int tag)
    {
        MPI_Status status;
        MPI_Recv(static_cast<void *>(data), size, type, src, tag, this->_comm, &status);
        return status;
    }

    template <typename T>
    MPI_Request ISend(const T *data, size_t size, MPI_Datatype type, int dst, int tag)
    {
        MPI_Request request;
        MPI_Isend(static_cast<const void *>(data), size, type, dst, tag, this->_comm, &request);
        return request;
    }

    template <typename T>
    MPI_Request IRecv(T *data, size_t size, MPI_Datatype type, int src, int tag)
    {
        MPI_Request request;
        MPI_Irecv(static_cast<void *>(data), size, type, src, tag, this->_comm, &request);
        return request;
    }

    template <typename T>
    auto Bcast(T *data, size_t size, MPI_Datatype type, int root)
    {
        return MPI_Bcast(static_cast<void *>(data), size, type, root, this->_comm);
    }

    template <typename T, typename U>
    auto Scatter(T *data, size_t ssize, MPI_Datatype stype, U *rdata, MPI_Datatype rtype, size_t rsize, int root)
    {
        return MPI_Scatter(data, ssize, stype, rdata, rsize, rtype, root, this->_comm);
    }

    template <typename T, typename U>
    auto Gather(T *data, size_t ssize, MPI_Datatype stype, U *rdata, MPI_Datatype rtype, size_t rsize, int root)
    {
        return MPI_Gather(data, ssize, stype, rdata, rsize, rtype, root, this->_comm);
    }

    template <typename T, typename U>
    auto AllGather(T *data, size_t ssize, MPI_Datatype stype, U *rdata, MPI_Datatype rtype, size_t rsize)
    {
        return MPI_Allgather(data, ssize, stype, rdata, rsize, rtype, this->_comm);
    }

    template <typename T>
    auto Allreduce(T *sbuff, T *rbuff, size_t size, MPI_Datatype type, MPI_Op op)
    {
        return MPI_Allreduce(sbuff, rbuff, size, type, op, this->_comm);
    }

    template <typename T>
    auto Reduce(T *sbuff, T *rbuff, size_t size, MPI_Datatype type, int root, MPI_Op op)
    {
        return Reduce(sbuff, rbuff, size, type, op, root, this->_comm);
    }

    // get_instance
    static driver *get_instance()
    {
        static driver instance;
        return &instance;
    }

private:

    // privates
    int _rank;
    int _size;
    bool _aware;
    MPI_Comm _comm;
};

    status_t apitest()
    {
        status_t status = SUCCESS;

        // get a MPI driver instance
        auto *driver = driver::get_instance();

        // initialize MPI
        driver->init(0, nullptr);

        // get rank and size
        int rank = driver->rank();
        int size = driver->size();

        int rankminus1 = (rank-1) < 0 ? size-1 : rank-1;
        int rankplus1 = (rank+1) % size;

        // check if we have at least 2 ranks
        if (size < 2)
        {
            std::cout << "FATAL: Need at least 2 ranks to run the apitest" << std::endl;
            return FAILURE;
        }
        else
        {
            if (!rank)
            {
                std::cout << "Running MPI apitest with " << size << " ranks" << std::endl << std::endl << std::flush;
            }
        }

        // to ensure above messages have been printed
        driver->barrier();

        // test message
        string_t msg = "testing...";
        const char *smessage = msg.c_str();
        char rmessage[256];

        // validate function for string
        auto validate = [&]()
        {
            driver->barrier();
            return strncmp(smessage, rmessage, msg.size());
        };

        // testing message printer
        auto test_message = [&](string_t &&msg)
        {
            if (!rank)
                std::cout << "Testing " << msg << ".." << std::endl << std::flush;
        };

        // status message printer
        auto status_message = [&](string_t &&msg, bool status)
        {
            if (!rank)
                std::cout << msg << " success: " << std::boolalpha << !status << std::endl << std::endl << std::flush;
        };

        // test a simple P2P ring exchange
        test_message("P2P");

        if (rank == 0)
        {
            driver->Send(smessage, msg.size(), MPI_CHAR, rankplus1, 0);
            driver->Recv(&rmessage[0], msg.size(), MPI_CHAR, rankminus1, 0);
        }
        else
        {
            driver->Recv(&rmessage[0], msg.size(), MPI_CHAR, rankminus1, 0);
            driver->Send(smessage, msg.size(), MPI_CHAR, rankplus1, 0);
        }

        // validate
        status = validate();

        if (!status)
        {
            auto &&sreq = driver->ISend(smessage, msg.size(), MPI_CHAR, rankplus1, 0);
            auto &&rreq = driver->IRecv(&rmessage[0], msg.size(), MPI_CHAR, rankminus1, 0);
            MPI_Request reqs[2] = {rreq, sreq};
            MPI_Status stats[2];
            MPI_Waitall(2, reqs, stats);
        }

        // validate and print status
        status = validate();

        status_message("P2P", status);

        // test a broadcast
        test_message("Bcast");

        if (!status)
        {
            if (!rank)
                driver->Bcast(msg.data(), msg.size(), MPI_CHAR, 0);
            else
                driver->Bcast(&rmessage[0], msg.size(), MPI_CHAR, 0);
        }

        // validate and print status

        status = validate();

        status_message("Bcast", status);

        // test allreduce
        test_message("Allreduce");

        if (!status)
        {
            int sum = -1;

            // allreduce
            MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM, driver->comm());
            status = !(sum == (size*(size-1))/2);
        }

        status_message("Allreduce", status);

        // test scatter
        test_message("Scatter");

        if (!status)
        {
            std::vector<int> v(size, -1);
            std::iota(v.begin(), v.end(), 0);

            int recv_1 = -1;

            // scatter
            driver->Scatter(v.data(), 1, MPI_INT, &recv_1,1, MPI_INT, 0);

            status = !(recv_1 == rank);
        }

        status_message("Scatter", status);

        // test all gather
        test_message("Allgather");

        // allgather
        if (!status)
        {
            std::vector<int> v(size, -1);
            std::iota(v.begin(), v.end(), 0);

            std::vector<int> recv(size, -1);
            MPI_Allgather(&rank, 1, MPI_INT, recv.data(), 1, MPI_INT, driver->comm());

            driver->barrier();

            for (int i = 0; i < size; i++)
                if (recv[i] != v[i])
                {
                    status = FAILURE;
                    break;
                }
        }

        status_message("Allgather", status);

        // test one way MPI communications
        test_message("MPI_Get");

        if (!status)
        {
            // one sided
            int win_buf = rank;
            MPI_Win window;
            MPI_Win_create(&win_buf, sizeof(int), sizeof(int), MPI_INFO_NULL, driver->comm(), &window);

            MPI_Win_fence(0, window);

            int rval = rank;

            // Fetch the value
            MPI_Get(&rval, 1, MPI_INT, rankplus1, 0, 1, MPI_INT, window);

            MPI_Win_fence(0, window);

            if (!(rval == rankplus1))
                status = FAILURE;

            status_message("MPI_Get", status);

            test_message("MPI_Put");

            if (!status)
            {
                rval = rank;

                MPI_Put(&rval, 1, MPI_INT, rankminus1, 0, 1, MPI_INT, window);

                MPI_Win_fence(0, window);

                status = !(win_buf == rankplus1);
            }

            MPI_Win_free(&window);
        }

        status_message("MPI_Put", status);

        // barrier to ensure we are done
        driver->barrier();

        // finalize MPI driver
        driver->finalize();

        return status;
    }

} // namespace mpi
} // namespace comm
} // namespace gcb

