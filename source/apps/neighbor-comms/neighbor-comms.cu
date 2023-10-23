/*
Copyright 2021, Evan Weinberg (eweinberg@nvidia.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "neighbor-comms.cuh"

// dummy "packing" kernel just to touch the memory
void __global__ packing_kernel(double* data, int size) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) return;

  data[tid] = static_cast<double>(tid);
}

int main(int argc, char** argv)
{

  // pre-process cmd info

  // expects 7 arguments: x, y, z partitions, message size for each comms
  if (argc != 8) {
    printf("ERROR: invalid number of inputs, expects seven\n");
  }

  // Initialize
  MPI_Init(&argc, &argv);

  int gpus_per_node = atoi(argv[1]);

  int x_decomp = atoi(argv[2]);
  int y_decomp = atoi(argv[3]);
  int z_decomp = atoi(argv[4]);

  int x_msg_size = atoi(argv[5]);
  x_msg_size = ((x_msg_size + sizeof(double) - 1) / sizeof(double)) * sizeof(double);
  int x_msg_doubles = x_msg_size / sizeof(double);

  int y_msg_size = atoi(argv[6]);
  y_msg_size = ((y_msg_size + sizeof(double) - 1) / sizeof(double)) * sizeof(double);
  int y_msg_doubles = y_msg_size / sizeof(double);

  int z_msg_size = atoi(argv[7]);
  z_msg_size = ((z_msg_size + sizeof(double) - 1) / sizeof(double)) * sizeof(double);
  int z_msg_doubles = z_msg_size / sizeof(double);

  bool x_part = (x_decomp > 1);
  bool y_part = (y_decomp > 1);
  bool z_part = (z_decomp > 1);

  // Various checks
  if (x_msg_size <= 0 && x_decomp > 1) {
    printf("Invalid x message size %d, expected > 0 because x is partitioned\n", x_msg_size); MPI_Finalize(); return -1;
  }
  if (x_decomp == 1) x_msg_size = 0;

  if (y_msg_size <= 0 && y_decomp > 1) {
    printf("Invalid y message size %d, expected > 0 because y is partitioned\n", y_msg_size);  MPI_Finalize(); return -1;
  }
  if (y_decomp == 1) y_msg_size = 0;

  if (z_msg_size <= 0 && z_decomp > 1) {
    printf("Invalid z message size %d, expected > 0 because z is partitioned\n", z_msg_size); MPI_Finalize(); return -1;
  }
  if (z_decomp == 1) z_msg_size = 0;

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_size != (x_decomp * y_decomp * z_decomp)) {
    printf("Number of MPI ranks %d != total decomposition size %d, exiting...", world_size, x_decomp*y_decomp*z_decomp);
    MPI_Finalize();
    return -1;
  }

  if (world_rank == 0) {
    printf("Decomposition { %d , %d , %d } Msg Size { %d , %d , %d }\n", x_decomp, y_decomp, z_decomp, x_msg_size, y_msg_size, z_msg_size);
  }

  // set the device
  checkCudaErrors(cudaSetDevice(world_rank % gpus_per_node));

  // create a stream, events for timing
  cudaStream_t stream;
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaStreamCreate(&stream);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // get my coordinates, x fastest
  int x_node = world_rank % x_decomp;
  int y_node = (world_rank / x_decomp) % y_decomp;
  int z_node = world_rank / (x_decomp * y_decomp);

  // get my neighbors
  int neighbors[6] = { ((x_node + 1) % x_decomp) + y_node * x_decomp + z_node * x_decomp * y_decomp,
                       ((x_node - 1 + x_decomp) % x_decomp) + y_node * x_decomp + z_node * x_decomp * y_decomp,
                       x_node + ((y_node + 1) % y_decomp) * x_decomp + z_node * x_decomp * y_decomp,
                       x_node + ((y_node - 1 + y_decomp) % y_decomp) * x_decomp + z_node * x_decomp * y_decomp,
                       x_node + y_node * x_decomp + ((z_node + 1) % z_decomp) * x_decomp * y_decomp,
                       x_node + y_node * x_decomp + ((z_node - 1 + z_decomp) % z_decomp) * x_decomp * y_decomp };

  // allocate buffers
  auto x_forward = std::unique_ptr<double,CudaMallocDeleter>(nullptr);
  auto x_backward = std::unique_ptr<double,CudaMallocDeleter>(nullptr);
  auto y_forward = std::unique_ptr<double,CudaMallocDeleter>(nullptr);
  auto y_backward = std::unique_ptr<double,CudaMallocDeleter>(nullptr);
  auto z_forward = std::unique_ptr<double,CudaMallocDeleter>(nullptr);
  auto z_backward = std::unique_ptr<double,CudaMallocDeleter>(nullptr);

  // receive buffer
  auto recv_buffer = std::unique_ptr<double,CudaMallocDeleter>(nullptr);

  if (x_part) {
    double* tmp;
    checkCudaErrors(cudaMalloc(&tmp, x_msg_size));
    x_forward.reset(tmp);
    checkCudaErrors(cudaMalloc(&tmp, x_msg_size));
    x_backward.reset(tmp);
  }

  if (y_part) {
    double* tmp;
    checkCudaErrors(cudaMalloc(&tmp, y_msg_size));
    y_forward.reset(tmp);
    checkCudaErrors(cudaMalloc(&tmp, y_msg_size));
    y_backward.reset(tmp);
  }

  if (z_part) {
    double* tmp;
    checkCudaErrors(cudaMalloc(&tmp, z_msg_size));
    z_forward.reset(tmp);
    checkCudaErrors(cudaMalloc(&tmp, z_msg_size));
    z_backward.reset(tmp);
  }

  if (x_part || y_part || z_part) {
    double* tmp;
    int max_buffer = (x_msg_size > y_msg_size) ? x_msg_size : y_msg_size;
    max_buffer = (max_buffer > z_msg_size) ? max_buffer : z_msg_size;
    checkCudaErrors(cudaMalloc(&tmp, max_buffer));
    recv_buffer.reset(tmp);
  }

  const int num_rep = 1000;
  if (world_rank == 0) { printf("Performing %d repetitions\n", num_rep); fflush(stdout); }

  MPI_Request request;

  // insert an event
  checkCudaErrors(cudaEventRecord(start, stream));

  for (int i = 0; i < num_rep; i++) {

    // X direction
    if (x_part) {

      ////////////////////////
      // Forward comm first //
      ////////////////////////

      // Prepare to receive from backwards rank
      MPI_Irecv(recv_buffer.get(), x_msg_doubles, MPI_DOUBLE,
                neighbors[XMINUS], 0, MPI_COMM_WORLD, &request);

      // packing kernel
      packing_kernel<<<(x_msg_doubles + 255) / 256, 256, 0, stream>>>(x_forward.get(), x_msg_doubles);
      checkCudaErrors(cudaStreamSynchronize(stream));

      // send forwards
      MPI_Send(x_forward.get(), x_msg_doubles, MPI_DOUBLE,
               neighbors[XPLUS], 0, MPI_COMM_WORLD);

      // wait
      MPI_Wait(&request, MPI_STATUS_IGNORE);

      ////////////////////////
      // Backward comm next //
      ////////////////////////

      // Prepare to receive from forwards rank
      MPI_Irecv(recv_buffer.get(), x_msg_doubles, MPI_DOUBLE,
                neighbors[XPLUS], 0, MPI_COMM_WORLD, &request);

      // packing kernel
      packing_kernel<<<(x_msg_doubles + 255) / 256, 256, 0, stream>>>(x_forward.get(), x_msg_doubles);
      checkCudaErrors(cudaStreamSynchronize(stream));

      // send backwards
      MPI_Send(x_backward.get(), x_msg_doubles, MPI_DOUBLE,
               neighbors[XMINUS], 0, MPI_COMM_WORLD);

      // wait
      MPI_Wait(&request, MPI_STATUS_IGNORE);

  }

    // Y direction
    if (y_part) {

      ////////////////////////
      // Forward comm first //
      ////////////////////////

      // Prepare to receive from backwards rank
      MPI_Irecv(recv_buffer.get(), y_msg_size / sizeof(double), MPI_DOUBLE,
                neighbors[YMINUS], 0, MPI_COMM_WORLD, &request);

      // packing kernel
      packing_kernel<<<(y_msg_doubles + 255) / 256, 256, 0, stream>>>(y_backward.get(), y_msg_doubles);
      checkCudaErrors(cudaStreamSynchronize(stream));

      // send forwards
      MPI_Send(y_forward.get(), y_msg_size / sizeof(double), MPI_DOUBLE,
               neighbors[YPLUS], 0, MPI_COMM_WORLD);

      // wait
      MPI_Wait(&request, MPI_STATUS_IGNORE);

      ////////////////////////
      // Backward comm next //
      ////////////////////////

      // Prepare to receive from forwards rank
      MPI_Irecv(recv_buffer.get(), y_msg_size / sizeof(double), MPI_DOUBLE,
                neighbors[YPLUS], 0, MPI_COMM_WORLD, &request);

      // packing kernel
      packing_kernel<<<(y_msg_doubles + 255) / 256, 256, 0, stream>>>(y_backward.get(), y_msg_doubles);
      checkCudaErrors(cudaStreamSynchronize(stream));

      // send backwards
      MPI_Send(y_backward.get(), y_msg_size / sizeof(double), MPI_DOUBLE,
               neighbors[YMINUS], 0, MPI_COMM_WORLD);

      // wait
      MPI_Wait(&request, MPI_STATUS_IGNORE);

    }

    // Z direction
    if (z_part) {

      ////////////////////////
      // Forward comm first //
      ////////////////////////

      // Prepare to receive from backwards rank
      MPI_Irecv(recv_buffer.get(), z_msg_size / sizeof(double), MPI_DOUBLE,
                neighbors[ZMINUS], 0, MPI_COMM_WORLD, &request);

      // packing kernel
      packing_kernel<<<(z_msg_doubles + 255) / 256, 256, 0, stream>>>(z_forward.get(), z_msg_doubles);
      checkCudaErrors(cudaStreamSynchronize(stream));

      // send forwards
      MPI_Send(z_forward.get(), z_msg_size / sizeof(double), MPI_DOUBLE,
               neighbors[ZPLUS], 0, MPI_COMM_WORLD);

      // wait
      MPI_Wait(&request, MPI_STATUS_IGNORE);

      ////////////////////////
      // Backward comm next //
      ////////////////////////

      // Prepare to receive from forwards rank
      MPI_Irecv(recv_buffer.get(), z_msg_size / sizeof(double), MPI_DOUBLE,
                neighbors[ZPLUS], 0, MPI_COMM_WORLD, &request);

      // packing kernel
      packing_kernel<<<(z_msg_doubles + 255) / 256, 256, 0, stream>>>(z_backward.get(), z_msg_doubles);
      checkCudaErrors(cudaStreamSynchronize(stream));

      // send backwards
      MPI_Send(z_backward.get(), z_msg_size / sizeof(double), MPI_DOUBLE,
               neighbors[ZMINUS], 0, MPI_COMM_WORLD);

      // wait
      MPI_Wait(&request, MPI_STATUS_IGNORE);

    }
  }

  checkCudaErrors(cudaDeviceSynchronize()); // to also sync any streams MPI is using

  checkCudaErrors(cudaEventRecord(stop, stream));
  checkCudaErrors(cudaEventSynchronize(stop));

  float milliseconds = 0.f;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  float times[world_size];
  MPI_Gather(&milliseconds, 1, MPI_FLOAT, &times, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  float t_avg = 0;
  float t2_avg = 0;
  float t_max = 0;
  float t_min = 1e50;
  if (world_rank == 0) {
    for (int i = 0; i < world_size; i++) {
      printf("Rank %d took %e milliseconds\n", i, times[i]);
      t_avg += times[i];
      t2_avg += times[i]*times[i];
      t_max = (t_max > times[i]) ? t_max : times[i];
      t_min = (t_min < times[i]) ? t_min : times[i];
    }

    printf("Avg %e\tStdDev %e\tMin %e\tMax %e\n", t_avg / world_size, sqrt(t2_avg / world_size - t_avg * t_avg / (world_size * world_size)), t_min, t_max);
  }

  printf("\n");

  // destroy the cuda stream, events
  checkCudaErrors(cudaStreamDestroy(stream));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  // Finalize the MPI environment.
  MPI_Finalize();

  return 0;
}

