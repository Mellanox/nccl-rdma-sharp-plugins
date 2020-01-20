/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_UTILS_H_
#define NCCL_UTILS_H_

#include "nccl.h"
#include "nccl_net.h"
#include "param.h"
#include <stdint.h>
#include <pthread.h>

#define NCCL_STATIC_ASSERT(_cond, _msg) \
    switch(0) {case 0:case (_cond):;}

#define MAXNAMESIZE 64
struct ncclIbDev {
  int device;
  uint8_t port;
  uint8_t link;
  uint8_t isSharpDev;
  struct ibv_context* context;
  char devName[MAXNAMESIZE];
};

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked DONTFORK
// and if they are shared, that could cause a crash in a child process
ncclResult_t ncclIbMalloc(void** ptr, size_t size);

ncclResult_t getHostName(char* hostname, int maxlen);
uint64_t getHostHash();
uint64_t getPidHash();

struct netIf {
  char prefix[64];
  int port;
};

int devCompare(const void *a, const void *b);
int parseStringList(const char* string, struct netIf* ifList, int maxList);
int matchIfList(const char* string, int port, struct netIf* ifList, int listSize);
int readFileNumber(long *value, const char *filename_fmt, ...);

#endif
