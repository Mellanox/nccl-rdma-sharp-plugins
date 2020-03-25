/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CORE_H_
#define NCCL_CORE_H_

#include "nccl.h"
#include "debug.h"

#include <stdint.h>
#include <stdlib.h>

#define MAX_REQUESTS 128
#define MAXNAMESIZE 64
#define MAX_IB_DEVS 16
#define IB_DEVICE_SYSFS_FMT "/sys/class/infiniband/%s/device/%s"

#define MIN(a, b) ((a)<(b)?(a):(b))
#define MAX(a, b) ((a)>(b)?(a):(b))

#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))
#define ROUNDUP(x, y) \
    (DIVUP((x), (y))*(y))

// Check CUDA calls
#define CUDACHECK(cmd) do {                                 \
    cudaError_t e = cmd;                                    \
    if( e != cudaSuccess ) {                                \
        WARN("Cuda failure '%s'", cudaGetErrorString(e));   \
        return ncclUnhandledCudaError;                      \
    }                                                       \
} while(0)

#include <errno.h>
// Check system calls
#define SYSCHECK(call, name) do { \
  int retval; \
  SYSCHECKVAL(call, name, retval); \
} while (0)

#define SYSCHECKVAL(call, name, retval) do { \
  SYSCHECKSYNC(call, name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed : %s", strerror(errno)); \
    return ncclSystemError; \
  } \
} while (0);

#define SYSCHECKSYNC(call, name, retval) do { \
  retval = call; \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    INFO(NCCL_ALL,"Call to " name " returned %s, retrying", strerror(errno)); \
  } else { \
    break; \
  } \
} while(0)

// Propagate errors up
#define NCCLCHECK(call) do { \
  ncclResult_t res = call; \
  if (res != ncclSuccess) { \
    /* Print the back trace*/ \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    return res; \
  } \
} while (0);

#define NCCLCHECKGOTO(call, res, label) do { \
  res = call; \
  if (res != ncclSuccess) { \
    /* Print the back trace*/ \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    goto label; \
  } \
} while (0);

struct ncclIbDev {
  int device;
  uint64_t guid;
  uint8_t port;
  uint8_t link;
  uint8_t isSharpDev;
  int speed;
  struct ibv_context* context;
  char devName[MAXNAMESIZE];
  char* pciPath;
  int realPort;
  int maxQp;
};

struct userIbDev {
  char devName[MAXNAMESIZE];
  uint16_t port_en;
};

#endif // end include guard
