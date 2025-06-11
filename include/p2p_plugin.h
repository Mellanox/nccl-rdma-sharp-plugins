/*************************************************************************
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2016-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_P2P_PLUGIN_H_
#define NCCL_P2P_PLUGIN_H_

#include <stdint.h>
#include <unistd.h>
#include <assert.h>

#include "nccl.h"
#include "net.h"
#include "ibvwrap.h"
#include "param.h"
#include "socket.h"
#include "utils.h"

#define MAXSUFFIXSIZE 16
#define MAXNAMESIZE  (64 + MAXSUFFIXSIZE)
#define NCCL_NET_IB_MAX_RECVS 8
// We need to support NCCL_NET_MAX_REQUESTS for each concurrent receive
#define MAX_REQUESTS (NCCL_NET_MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS)
//static_assert(MAX_REQUESTS <= 256, "request id are encoded in wr_id and we need up to 8 requests ids per completion");
#define IB_DEVICE_SYSFS_FMT "/sys/class/infiniband/%s/device/%s"

#define NCCL_IB_LLSTR(ll) (((ll) == IBV_LINK_LAYER_INFINIBAND) ? "IB" : (((ll) == IBV_LINK_LAYER_ETHERNET) ? "RoCE" : "UNSPECIFIED"))

typedef enum nccl_p2p_plugin {
  NCCL_P2P_IB,
  NCCL_P2P_UCX,
  NCCL_P2P_UCX_RMA,
  NCCL_P2P_UCX_UCT,
  NCCL_P2P_UCX_UCT_RD,
  NCCL_P2P_LAST
} nccl_p2p_plugin_t;

struct ncclIbMr {
  uintptr_t addr;
  size_t pages;
  int refs;
  struct ibv_mr *mr;
};

struct ncclIbMrCache {
  struct ncclIbMr *slots;
  int capacity, population;
};

#define NCCL_IB_MAX_DEVS_PER_NIC 4
#define MAX_MERGED_DEV_NAME (MAXNAMESIZE*NCCL_IB_MAX_DEVS_PER_NIC)+NCCL_IB_MAX_DEVS_PER_NIC
typedef struct ncclIbMergedDev {
  ncclNetVDeviceProps_t vProps;
  int speed;
  char devName[MAX_MERGED_DEV_NAME]; // Up to NCCL_IB_MAX_DEVS_PER_NIC * name size, and a character for each '+'
} __attribute__((aligned(64))) ncclIbMergedDev;

struct ncclIbStats {
  int fatalErrorCount;
};

struct ncclIbRequest {
  struct ncclIbNetCommBase* base;
  int type;
  struct ncclSocket* sock;
  int events[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbNetCommDevBase* devBases[NCCL_IB_MAX_DEVS_PER_NIC];
  int nreqs;
  union {
    struct {
      int size;
      void* data;
      uint32_t lkeys[NCCL_IB_MAX_DEVS_PER_NIC];
      int offset;
    } send;
    struct {
      int* sizes;
    } recv;
  };
};

// Retain local RoCE address for error logging
struct ncclIbGidInfo {
  uint8_t link_layer;
  union ibv_gid localGid;
  int32_t localGidIndex;
};

typedef struct ncclIbNetCommDevBase {
  int ibDevN;
  struct ibv_pd* pd;
  struct ibv_cq* cq;
  uint64_t pad[2];
  struct ncclIbGidInfo gidInfo;
} ncclIbNetCommDevBase;

enum ncclIbProvider {
  IB_PROVIDER_NONE = 0,
  IB_PROVIDER_MLX5 = 1,
  IB_PROVIDER_MAX = 2,
};
typedef struct ncclIbDev {
  pthread_mutex_t lock;
  int      device;
  uint64_t guid;
  uint8_t portNum;
  uint8_t  link;
  uint8_t  isSharpDev;
  int      speed;
  struct   ibv_context* context;
  int      pdRefs;
  struct ibv_pd*  pd;
  char     devName[MAXNAMESIZE];
  char     *pciPath;
  char* virtualPciPath;
  int      realPort;
  int      maxQp;
  float latency;
  struct   ncclIbMrCache mrCache;
  int ar; // ADAPTIVE_ROUTING
  struct ibv_port_attr portAttr;
  struct ncclIbStats stats;
  int dmaBufSupported;
  enum ncclIbProvider ibProvider;
  union {
    struct {
      int dataDirect;
    } mlx5;
  } capsProvider;
} __attribute__((aligned(64))) ncclIbDev;


#define MAX_IB_DEVS  32
#define MAX_IB_VDEVS MAX_IB_DEVS*8
extern struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_VDEVS];
extern struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
/* Detect whether GDR can work on a given NIC with the current CUDA device
 * Returns :
 * ncclSuccess : GDR works
 * ncclSystemError : no module or module loaded but not supported by GPU */
ncclResult_t nccl_p2p_gdr_support();

ncclResult_t nccl_p2p_dmabuf_support(int dev);

ncclResult_t nccl_p2p_ib_pci_path(ncclIbDev *devs, int num_devs, char* dev_name, char** path, int* real_port);

ncclResult_t nccl_p2p_ib_get_properties(ncclIbDev *devs, int ncclNMergedIbDevs, int dev, ncclNetProperties_t* props);

ncclResult_t nccl_p2p_ib_init(int *nDevs, int *nmDevs, ncclIbDev *ncclIbDevs, char *ncclIbIfName, union ncclSocketAddress *ncclIbIfAddr,
                              pthread_t *ncclIbAsyncThread, ncclDebugLogger_t logFunction);

/* Convert value returtned by ibv_query_port to actual link width */
int nccl_p2p_ib_width(int width);

/* Convert value returtned by ibv_query_port to actual link speed */
int nccl_p2p_ib_speed(int speed);

int64_t ncclParamSharpMaxComms();

int64_t ncclParamIbMergeVfs();

int64_t ncclParamIbMergeNics();

int ncclIbRelaxedOrderingCapable(void);

nccl_p2p_plugin_t nccl_p2p_get_plugin_type();

ncclResult_t ncclIbStatsInit(struct ncclIbStats* stat);

ncclResult_t ncclIbMakeVDeviceInternal(int* d, ncclNetVDeviceProps_t* props, int nDevs, int *nmDevs);

#endif
