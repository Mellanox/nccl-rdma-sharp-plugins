/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 * Copyright (C) 2019-2020, Mellanox Technologies Ltd. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <strings.h>

#include "nccl.h"
#include "nccl_net.h"

extern ncclNet_t ucxPlugin;
extern ncclNet_t ibPlugin;
ncclDebugLogger_t pluginLogFunction;

ncclResult_t pluginInit(ncclDebugLogger_t logFunction);

ncclNet_t NCCL_PLUGIN_SYMBOL = {
  "NCCL RDMA SHARP Plugin",
  pluginInit,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL
};


ncclResult_t pluginInit(ncclDebugLogger_t logFunction){
  pluginLogFunction = logFunction;
  NCCL_PLUGIN_SYMBOL = ibPlugin;
#ifdef HAVE_UCX_PLUGIN
  const char *p2pLayer = getenv("NCCL_PLUGIN_P2P");
  if ((p2pLayer != NULL) && !strcasecmp(p2pLayer, "ucx"))  NCCL_PLUGIN_SYMBOL = ucxPlugin;
#endif
  return NCCL_PLUGIN_SYMBOL.init(logFunction);
}
