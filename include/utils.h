/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_UTILS_H_
#define NCCL_UTILS_H_

#include "nccl.h"
#include <stdint.h>

#define NCCL_STATIC_ASSERT(_cond, _msg) \
    switch(0) {case 0:case (_cond):;}

ncclResult_t ncclIbMalloc(void** ptr, size_t size);
ncclResult_t ncclRealloc(void** ptr, size_t old_size, size_t new_size);
ncclResult_t getHostName(char* hostname, int maxlen);
uint64_t getHostHash();
uint64_t getPidHash();

struct netIf {
  char prefix[64];
  int port;
};

int parseStringList(const char* string, struct netIf* ifList, int maxList);
int matchIfList(const char* string, int port, struct netIf* ifList, int listSize, int matchExact);
const char *get_plugin_lib_path();

#endif
