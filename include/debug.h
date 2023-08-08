/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEBUG_H_
#define NCCL_DEBUG_H_

#include "core.h"

#include <stdio.h>

#include <sys/syscall.h>
#include <limits.h>
#include <string.h>
#include <pthread.h>
#include "net.h"

// Conform to pthread and NVTX standard
#define NCCL_THREAD_NAMELEN 16

extern pthread_mutex_t ncclDebugLock;

extern ncclDebugLogger_t pluginLogFunction;

#define WARN(...) pluginLogFunction(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) pluginLogFunction(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)

#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...) pluginLogFunction(NCCL_LOG_TRACE, (FLAGS), __func__, __LINE__, __VA_ARGS__)
#else
#define TRACE(...)
#endif

void ncclSetThreadName(pthread_t thread, const char *fmt, ...);

#endif
