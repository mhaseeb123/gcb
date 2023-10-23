#include "cuda/driver.hpp"
#include "commlibs/mpi/driver.hpp"

namespace gcb
{
namespace algs
{

template <typename T, typename U>
status_t flood_mesh(T *data, settings_t &settings)
{
    status_t status = SUCCESS;

    auto &comm_drvr = U::get_instance();
    comm_drvr->init(0, nullptr);

    auto &&rank = comm_drvr->rank();
    auto &&size = comm_drvr->size();

    auto tx = [&]() {
        // transmit flood
        for (int i = 0; i < size; i++)
        {
            if (i == rank)
                continue;
            comm_drvr->send(data, settings.msg_size * sizeof(T), MPI_CHAR, i, 0xF100D);
        }
    };

    auto rx = [&]() {
        // receive flood
        for (int i = 0; i < size; i++)
        {
            if (i == rank)
                continue;

            comm_drvr->recv(data, settings.msg_size * sizeof(T), MPI_CHAR, i, 0xF100D);
        }
    };

    return status;
}

} // namespace algs
} // namespace gcb


