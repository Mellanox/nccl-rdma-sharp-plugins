/*************************************************************************
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CORE_H_
#define NCCL_CORE_H_

#include "nccl.h"
#include "debug.h"

#include <stdint.h>
#include <stdlib.h>

#define MIN(a, b) ((a)<(b)?(a):(b))
#define MAX(a, b) ((a)>(b)?(a):(b))

#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))
#define ROUNDUP(x, y) \
    (DIVUP((x), (y))*(y))

// Check CUDA calls
#define CUDACHECK(cmd) do {                                 \
    cudaError_t err = cmd;                                  \
    if( err != cudaSuccess ) {                              \
        WARN("Cuda failure '%s'", cudaGetErrorString(err)); \
        return ncclUnhandledCudaError;                      \
    }                                                       \
} while(false)

#define CUDACHECKGOTO(cmd, RES, label) do {                 \
    cudaError_t err = cmd;                                  \
    if( err != cudaSuccess ) {                              \
        WARN("Cuda failure '%s'", cudaGetErrorString(err)); \
        RES = ncclUnhandledCudaError;                       \
        goto label;                                         \
    }                                                       \
} while(false)

// Report failure but clear error and continue
#define CUDACHECKIGNORE(cmd) do {  \
    cudaError_t err = cmd;         \
    if( err != cudaSuccess ) {     \
        INFO(NCCL_ALL,"%s:%d Cuda failure '%s'", __FILE__, __LINE__, cudaGetErrorString(err)); \
        (void) cudaGetLastError(); \
    }                              \
} while(false)

#include <errno.h>
// Check system calls
#define SYSCHECK(statement, name) do { \
  int retval; \
  SYSCHECKSYNC((statement), name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed: %s", strerror(errno)); \
    return ncclSystemError; \
  } \
} while (false)

#define SYSCHECKSYNC(statement, name, retval) do { \
  retval = (statement); \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    INFO(NCCL_ALL,"Call to " name " returned %s, retrying", strerror(errno)); \
  } else { \
    break; \
  } \
} while(true)

#define SYSCHECKGOTO(statement, name, RES, label) do { \
  int retval; \
  SYSCHECKSYNC((statement), name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed: %s", strerror(errno)); \
    RES = ncclSystemError; \
    goto label; \
  } \
} while (0)

// Pthread calls don't set errno and never return EINTR.
#define PTHREADCHECK(statement, name) do { \
  int retval = (statement); \
  if (retval != 0) { \
    WARN("Call to " name " failed: %s", strerror(retval)); \
    return ncclSystemError; \
  } \
} while (0)

#define PTHREADCHECKGOTO(statement, name, RES, label) do { \
  int retval = (statement); \
  if (retval != 0) { \
    WARN("Call to " name " failed: %s", strerror(retval)); \
    RES = ncclSystemError; \
    goto label; \
  } \
} while (0)


#define NEQCHECK(statement, value) do {   \
  if ((statement) != value) {             \
    /* Print the back trace*/             \
    INFO(NCCL_ALL,"%s:%d -> %d (%s)", __FILE__, __LINE__, ncclSystemError, strerror(errno));    \
    return ncclSystemError;     \
  }                             \
} while (0)

#define NEQCHECKGOTO(statement, value, RES, label) do { \
  if ((statement) != value) { \
    /* Print the back trace*/ \
    RES = ncclSystemError;    \
    INFO(NCCL_ALL,"%s:%d -> %d (%s)", __FILE__, __LINE__, RES, strerror(errno));    \
    goto label; \
  } \
} while (0)

#define EQCHECK(statement, value) do {    \
  if ((statement) == value) {             \
    /* Print the back trace*/             \
    INFO(NCCL_ALL,"%s:%d -> %d (%s)", __FILE__, __LINE__, ncclSystemError, strerror(errno));    \
    return ncclSystemError;     \
  }                             \
} while (0)

#define EQCHECKGOTO(statement, value, RES, label) do { \
  if ((statement) == value) { \
    /* Print the back trace*/ \
    RES = ncclSystemError;    \
    INFO(NCCL_ALL,"%s:%d -> %d (%s)", __FILE__, __LINE__, RES, strerror(errno));    \
    goto label; \
  } \
} while (0)

// Propagate errors up
#define NCCLCHECK(call) do { \
  ncclResult_t RES = call; \
  if (RES != ncclSuccess && RES != ncclInProgress) { \
    /* Print the back trace*/ \
    return RES; \
  } \
} while (0)

#define NCCLCHECKGOTO(call, RES, label) do { \
  RES = call; \
  if (RES != ncclSuccess && RES != ncclInProgress) { \
    /* Print the back trace*/ \
    goto label; \
  } \
} while (0)

#define NCCLWAIT(call, cond, abortFlagPtr) do {         \
  volatile uint32_t* tmpAbortFlag = (abortFlagPtr);     \
  ncclResult_t RES = call;                \
  if (RES != ncclSuccess && RES != ncclInProgress) {               \
    return ncclInternalError;             \
  }                                       \
  if (tmpAbortFlag) NEQCHECK(*tmpAbortFlag, 0); \
} while (!(cond))

#define NCCLWAITGOTO(call, cond, abortFlagPtr, RES, label) do { \
  volatile uint32_t* tmpAbortFlag = (abortFlagPtr);             \
  RES = call;                             \
  if (RES != ncclSuccess && RES != ncclInProgress) {               \
    goto label;                           \
  }                                       \
  if (tmpAbortFlag) NEQCHECKGOTO(*tmpAbortFlag, 0, RES, label); \
} while (!(cond))

#define NCCLCHECKTHREAD(a, args) do { \
  if (((args)->ret = (a)) != ncclSuccess && (args)->ret != ncclInProgress) { \
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, (args)->ret); \
    return args; \
  } \
} while(0)

#define CUDACHECKTHREAD(a) do { \
  if ((a) != cudaSuccess) { \
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, args->ret); \
    args->ret = ncclUnhandledCudaError; \
    return args; \
  } \
} while(0)

#endif
