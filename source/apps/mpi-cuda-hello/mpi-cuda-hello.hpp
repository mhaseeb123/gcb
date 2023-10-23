#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <sched.h>
#include <iostream>

int find_gpus(void)
{
    int ngpu;
    cudaGetDeviceCount(&ngpu);
    return ngpu;
}

void gpu_pci_id(char* device_id, int device_num)
{
    int len = 32;
    int ret = cudaDeviceGetPCIBusId(device_id, len, device_num) ;
    if (!ret)
        return;
    else
    {
        printf("Could not get the gpu-id. Error code = %d\n",ret);
        exit(1);
    }
}