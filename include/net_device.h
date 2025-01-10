/*************************************************************************
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_DEVICE_H_
#define NET_DEVICE_H_

#define NCCL_NET_DEVICE_INVALID_VERSION      0x0
#define NCCL_NET_MTU_SIZE                    4096

// Arbitrary version number - A given NCCL build will only be compatible with a single device networking plugin
// version. NCCL will check the supplied version number from net->getProperties() and compare to its internal version.
#define NCCL_NET_DEVICE_UNPACK_VERSION 0x7

typedef enum {NCCL_NET_DEVICE_HOST=0, NCCL_NET_DEVICE_UNPACK=1} ncclNetDeviceType;

typedef struct {
  ncclNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;            // Version number for network offload
  void* handle;
  size_t size;
  int needsProxyProgress;
} ncclNetDeviceHandle_v7_t;

typedef ncclNetDeviceHandle_v7_t ncclNetDeviceHandle_v8_t;
typedef ncclNetDeviceHandle_v8_t ncclNetDeviceHandle_v9_t;
typedef ncclNetDeviceHandle_v9_t ncclNetDeviceHandle_t;

#endif
