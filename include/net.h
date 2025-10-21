/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef NCCL_NET_H_
#define NCCL_NET_H_

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include "net_device.h"

#define NCCL_NET_HANDLE_MAXSIZE 128
//Maximum value NCCL can accept for maxP2pBytes and maxCollBytes net properties
#define NCCL_MAX_NET_SIZE_BYTES (1*1024*1024*1024*1024L)
#define MAX_COLLNET_SIZE (512*1024*1024L)
#define NCCL_NET_OPTIONAL_RECV_COMPLETION 0x1

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2
#define NCCL_PTR_DMABUF 0x4

#define NCCL_NET_MR_FLAG_FORCE_SO (1 << 0)
#define NCCL_NET_SIGNAL_OP_INC 0x1
#define NCCL_NET_SIGNAL_OP_ADD 0x2

// Maximum number of requests per comm object
#define NCCL_NET_MAX_REQUESTS 8
#define NCCL_NET_MAX_DEVS_PER_NIC 4

typedef enum {NCCL_LOG_NONE=0, NCCL_LOG_VERSION=1, NCCL_LOG_WARN=2, NCCL_LOG_INFO=3, NCCL_LOG_ABORT=4, NCCL_LOG_TRACE=5} ncclDebugLogLevel;
typedef enum {NCCL_INIT=1, NCCL_COLL=2, NCCL_P2P=4, NCCL_SHM=8, NCCL_NET=16, NCCL_GRAPH=32, NCCL_TUNING=64, NCCL_ENV=128, NCCL_ALLOC=256, NCCL_CALL=512, NCCL_ALL=~0} ncclDebugLogSubSys;

typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);
typedef ncclResult_t (*ncclProfilerCallback_t)(void** eHandle, int type, void* phandle, int64_t pluginId, void* extData);

#include "net_v11.h"
#include "net_v10.h"
#include "net_v9.h"
#include "net_v8.h"
#include "net_v7.h"
#include "net_v6.h"

typedef ncclNet_v11_t ncclNet_t;
typedef ncclCollNet_v11_t ncclCollNet_t;
typedef ncclNetProperties_v11_t ncclNetProperties_t;
typedef ncclNetVDeviceProps_v11_t ncclNetVDeviceProps_t;
typedef ncclNetCommConfig_v11_t ncclNetCommConfig_t;
typedef ncclNetAttr_v11_t ncclNetAttr_t;

#endif // end include guard
