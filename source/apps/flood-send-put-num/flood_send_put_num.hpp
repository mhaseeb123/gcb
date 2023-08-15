#pragma once

//#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <sys/time.h>
#include <string.h>
#include <inttypes.h>
#include <sched.h>
#include "gcb.hpp"
#include "commlibs/mpi/driver.hpp"
#include <sys/syscall.h>
#include <utmpx.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>