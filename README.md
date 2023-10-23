# GPU-GPU Communications & Benchmarks (GCB)
GCB develops a suite of apps and kernels to demonstrate, study (and optimize) GPU-GPU communications using new technologies/features in MPI, NCCL, NVSHMEM, and C++.

# Dependencies

* MPI
* NVHPC
* C++
* NCCL
* NVSHMEM
* argparse
* optional: magic_enum

# Build

## Perlmutter

```bash
# load modules
ml PrgEnv-nvhpc ; ml nvhpc/23.1 ; ml cmake

# build
mkdir build ; cd build ; cmake .. -DCMAKE_CUDA_HOST_COMPILER=$(which CC) [OTHER OPTIONS] ; make -j <JOBS>
```

# Run

```bash
srun -n <RANKS> -N <NODES> -C <PARTITION> -G <GPUS> -t <TIME> -A <ACCT> --ntasks-per-node=<> --gpus-per-node=<> [OTHER SLURM OPTIONS] build/apps/<appname>/<appname> [ARGS]
```

# Contact
Please open an issue or reach out at: mhaseeb@lbl.gov

# Contributors
Muhammad Haseeb <a href="https://github.com/mhaseeb123"><img alt="Twitter" src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" height=15>
