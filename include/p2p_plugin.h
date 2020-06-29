/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 * Copyright (C) 2019-2020, Mellanox Technologies Ltd. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_P2P_PLUGIN_H_
#define NCCL_P2P_PLUGIN_H_

#include <stdint.h>
#include <unistd.h>

#include "nccl.h"
#include "nccl_net.h"
#include "ibvwrap.h"
#include "param.h"
#include "socket.h"
#include "utils.h"

typedef enum nccl_p2p_plugin {
  NCCL_P2P_IB,
  NCCL_P2P_UCX,
  NCCL_P2P_UCX_RMA,
  NCCL_P2P_LAST
} nccl_p2p_plugin_t;

typedef struct ncclIbDev {
  int      device;
  uint64_t guid;
  uint8_t  port;
  uint8_t  link;
  uint8_t  isSharpDev;
  int      speed;
  struct   ibv_context* context;
  char     devName[MAXNAMESIZE];
  char     *pciPath;
  int      realPort;
  int      maxQp;
} nccl_ib_dev_t;

/* Detect whether GDR can work on a given NIC with the current CUDA device
 * Returns :
 * ncclSuccess : GDR works
 * ncclSystemError : no module or module loaded but not supported by GPU */
ncclResult_t nccl_p2p_gdr_support(int dev);

ncclResult_t nccl_p2p_ib_pci_path(nccl_ib_dev_t *devs, int num_devs, char* dev_name, char** path, int* real_port);

ncclResult_t nccl_p2p_ib_get_properties(nccl_ib_dev_t *devs, int dev, ncclNetProperties_t* props);

ncclResult_t nccl_p2p_ib_init(int *num_devs, nccl_ib_dev_t *ncclIbDevs, char *ncclIbIfName, union socketAddress *ncclIbIfAddr, pthread_t *ncclIbAsyncThread, ncclDebugLogger_t logFunction);

/* Convert value returtned by ibv_query_port to actual link width */
int nccl_p2p_ib_width(int width);

/* Convert value returtned by ibv_query_port to actual link speed */
int nccl_p2p_ib_speed(int speed);

int64_t ncclParamSharpMaxComms();

nccl_p2p_plugin_t nccl_p2p_get_plugin_type();

#endif
