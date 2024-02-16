/*************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define MAXNAMESIZE 64
#define NCCL_NET_IB_MAX_RECVS 8
// We need to support NCCL_NET_MAX_REQUESTS for each concurrent receive
#define MAX_REQUESTS (NCCL_NET_MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS)
//static_assert(MAX_REQUESTS <= 256, "request id are encoded in wr_id and we need up to 8 requests ids per completion");
#define IB_DEVICE_SYSFS_FMT "/sys/class/infiniband/%s/device/%s"

typedef enum nccl_p2p_plugin {
  NCCL_P2P_IB,
  NCCL_P2P_UCX,
  NCCL_P2P_UCX_RMA,
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

#define NCCL_IB_MAX_DEVS_PER_NIC 2
#define MAX_MERGED_DEV_NAME (MAXNAMESIZE*NCCL_IB_MAX_DEVS_PER_NIC)+NCCL_IB_MAX_DEVS_PER_NIC
struct ncclIbMergedDev {
  int ndevs;
  int devs[NCCL_IB_MAX_DEVS_PER_NIC]; // Points to an index in ncclIbDevs
  int speed;
  char devName[MAX_MERGED_DEV_NAME]; // Up to NCCL_IB_MAX_DEVS_PER_NIC * name size, and a character for each '+'
} __attribute__((aligned(64)));

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
};

typedef struct ncclIbNetCommDevBase {
  int ibDevN;
  struct ibv_pd* pd;
  struct ibv_cq* cq;
  uint64_t pad[1];
  struct ncclIbGidInfo gidInfo;
} ncclIbNetCommDevBase;

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
  int      realPort;
  int      maxQp;
  struct   ncclIbMrCache mrCache;
  int ar; // ADAPTIVE_ROUTING
  struct ibv_port_attr portAttr;
} __attribute__((aligned(64))) ncclIbDev;


#define MAX_IB_DEVS 32
extern struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_DEVS];
extern struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
/* Detect whether GDR can work on a given NIC with the current CUDA device
 * Returns :
 * ncclSuccess : GDR works
 * ncclSystemError : no module or module loaded but not supported by GPU */
ncclResult_t nccl_p2p_gdr_support();

ncclResult_t nccl_p2p_dmabuf_support(int dev);

ncclResult_t nccl_p2p_ib_pci_path(ncclIbDev *devs, int num_devs, char* dev_name, char** path, int* real_port);

ncclResult_t nccl_p2p_ib_get_properties(ncclIbDev *devs, int dev, ncclNetProperties_t* props);

ncclResult_t nccl_p2p_ib_init(int *num_devs, ncclIbDev *ncclIbDevs, char *ncclIbIfName, union ncclSocketAddress *ncclIbIfAddr, pthread_t *ncclIbAsyncThread, ncclDebugLogger_t logFunction);

/* Convert value returtned by ibv_query_port to actual link width */
int nccl_p2p_ib_width(int width);

/* Convert value returtned by ibv_query_port to actual link speed */
int nccl_p2p_ib_speed(int speed);

int64_t ncclParamSharpMaxComms();

int64_t ncclParamIbMergeVfs();

int64_t ncclParamIbMergeNics();

int ncclIbRelaxedOrderingCapable(void);

nccl_p2p_plugin_t nccl_p2p_get_plugin_type();

#endif
