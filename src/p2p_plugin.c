/*************************************************************************
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2016-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <strings.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/time.h>

#include "debug.h"
#include "p2p_plugin.h"

#ifdef HAVE_UCX_PLUGIN
extern ncclNet_v10_t ucxPlugin_v10;
extern ncclNet_v9_t ucxPlugin_v9;
extern ncclNet_v8_t ucxPlugin_v8;
extern ncclNet_v7_t ucxPlugin_v7;
extern ncclNet_v6_t ucxPlugin_v6;
extern ncclNet_v5_t ucxPlugin_v5;

extern ncclNet_v10_t ucxRmaPlugin_v10;
extern ncclNet_v9_t ucxRmaPlugin_v9;
extern ncclNet_v8_t ucxRmaPlugin_v8;
extern ncclNet_v7_t ucxRmaPlugin_v7;
extern ncclNet_v6_t ucxRmaPlugin_v6;
extern ncclNet_v5_t ucxRmaPlugin_v5;

extern ncclNet_v10_t ucxUctPlugin_v10;
extern ncclNet_v9_t ucxUctPlugin_v9;
extern ncclNet_v8_t ucxUctPlugin_v8;
extern ncclNet_v7_t ucxUctPlugin_v7;
extern ncclNet_v6_t ucxUctPlugin_v6;
extern ncclNet_v5_t ucxUctPlugin_v5;

extern ncclNet_v10_t ucxUctRdPlugin_v10;
extern ncclNet_v9_t ucxUctRdPlugin_v9;
extern ncclNet_v8_t ucxUctRdPlugin_v8;
extern ncclNet_v7_t ucxUctRdPlugin_v7;
extern ncclNet_v6_t ucxUctRdPlugin_v6;
extern ncclNet_v5_t ucxUctRdPlugin_v5;
#endif

extern ncclNet_v10_t ibPlugin_v10;
extern ncclNet_v9_t ibPlugin_v9;
extern ncclNet_v8_t ibPlugin_v8;
extern ncclNet_v7_t ibPlugin_v7;
extern ncclNet_v6_t ibPlugin_v6;
extern ncclNet_v5_t ibPlugin_v5;
pthread_mutex_t nccl_p2p_lock = PTHREAD_MUTEX_INITIALIZER;

ncclDebugLogger_t pluginLogFunction;


const char* ibProviderName[] = {
  "None",
  "Mlx5",
};

#ifdef HAVE_SHARP_PLUGIN
extern int ncclNSharpDevs;
#else
/* In case sharp plugin is not there just define this variable locally to make code cleaner */
int ncclNSharpDevs;
#endif
extern int ncclIbRelaxedOrderingEnabled;
NCCL_PARAM(SharpMaxComms, "SHARP_MAX_COMMS", 1);
NCCL_PARAM(IbAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2);
NCCL_PARAM(IbDataDirect,"IB_DATA_DIRECT",1);

ncclResult_t pluginInit_v10(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction);
ncclResult_t pluginInit_v9(ncclDebugLogger_t logFunction);
ncclResult_t pluginInit_v8(ncclDebugLogger_t logFunction);
ncclResult_t pluginInit_v7(ncclDebugLogger_t logFunction);
ncclResult_t pluginInit_v6(ncclDebugLogger_t logFunction);
ncclResult_t pluginInit_v5(ncclDebugLogger_t logFunction);

ncclNet_v10_t ncclNetPlugin_v10 = {
  "NCCL RDMA Plugin v10",
  pluginInit_v10,
};

ncclNet_v9_t ncclNetPlugin_v9 = {
  "NCCL RDMA Plugin v9",
  pluginInit_v9,
};

ncclNet_v8_t ncclNetPlugin_v8 = {
  "NCCL RDMA Plugin v8",
  pluginInit_v8,
};

ncclNet_v7_t ncclNetPlugin_v7 = {
  "NCCL RDMA Plugin v7",
  pluginInit_v7,
};

ncclNet_v6_t ncclNetPlugin_v6 = {
  "NCCL RDMA Plugin v6",
  pluginInit_v6,
};


ncclNet_v5_t ncclNetPlugin_v5 = {
  "NCCL RDMA Plugin v5",
  pluginInit_v5,
};


static nccl_p2p_plugin_t p2p_plugin = NCCL_P2P_LAST;

static int nccl_p2p_is_uct_plugin(nccl_p2p_plugin_t plugin) {
  return (plugin == NCCL_P2P_UCX_UCT) || (plugin == NCCL_P2P_UCX_UCT_RD);
}

static void pluginSetup()
{
  p2p_plugin = NCCL_P2P_IB;
  const char *plugin_path = get_plugin_lib_path();
  if (plugin_path != NULL) {
    INFO(NCCL_INIT|NCCL_NET, "Plugin Path : %s", plugin_path);;
  }

  const char *p2p_layer = getenv("NCCL_PLUGIN_P2P");
  if (p2p_layer != NULL) {
    if (!strcasecmp(p2p_layer, "ib")) p2p_plugin = NCCL_P2P_IB;
#ifdef HAVE_UCX_PLUGIN
    else if (!strcasecmp(p2p_layer, "ucx")) p2p_plugin = NCCL_P2P_UCX;
    else if (!strcasecmp(p2p_layer, "ucx_rma")) p2p_plugin = NCCL_P2P_UCX_RMA;
    else if (!strcasecmp(p2p_layer, "ucx_uct")) p2p_plugin = NCCL_P2P_UCX_UCT;
    else if (!strcasecmp(p2p_layer, "ucx_uct_read")) p2p_plugin = NCCL_P2P_UCX_UCT_RD;
#endif
    else {
      WARN("Invalid value %s for NCCL_PLUGIN_P2P, using default", p2p_layer);
    }
  }
  switch (p2p_plugin) {
#ifdef HAVE_UCX_PLUGIN
    case NCCL_P2P_UCX:
      ncclNetPlugin_v10 = ucxPlugin_v10;
      ncclNetPlugin_v9 = ucxPlugin_v9;
      ncclNetPlugin_v8 = ucxPlugin_v8;
      ncclNetPlugin_v7 = ucxPlugin_v7;
      ncclNetPlugin_v6 = ucxPlugin_v6;
      ncclNetPlugin_v5 = ucxPlugin_v5;
      break;
    case NCCL_P2P_UCX_RMA:
      ncclNetPlugin_v10 = ucxRmaPlugin_v10;
      ncclNetPlugin_v9 = ucxRmaPlugin_v9;
      ncclNetPlugin_v8 = ucxRmaPlugin_v8;
      ncclNetPlugin_v7 = ucxRmaPlugin_v7;
      ncclNetPlugin_v6 = ucxRmaPlugin_v6;
      ncclNetPlugin_v5 = ucxRmaPlugin_v5;
      break;
    case NCCL_P2P_UCX_UCT:
      ncclNetPlugin_v10 = ucxUctPlugin_v10;
      ncclNetPlugin_v9 = ucxUctPlugin_v9;
      ncclNetPlugin_v8 = ucxUctPlugin_v8;
      ncclNetPlugin_v7 = ucxUctPlugin_v7;
      ncclNetPlugin_v6 = ucxUctPlugin_v6;
      ncclNetPlugin_v5 = ucxUctPlugin_v5;
      break;
    case NCCL_P2P_UCX_UCT_RD:
      ncclNetPlugin_v10 = ucxUctRdPlugin_v10;
      ncclNetPlugin_v9 = ucxUctRdPlugin_v9;
      ncclNetPlugin_v8 = ucxUctRdPlugin_v8;
      ncclNetPlugin_v7 = ucxUctRdPlugin_v7;
      ncclNetPlugin_v6 = ucxUctRdPlugin_v6;
      ncclNetPlugin_v5 = ucxUctRdPlugin_v5;
      break;
#endif
    default:
      ncclNetPlugin_v10 = ibPlugin_v10;
      ncclNetPlugin_v9 = ibPlugin_v9;
      ncclNetPlugin_v8 = ibPlugin_v8;
      ncclNetPlugin_v7 = ibPlugin_v7;
      ncclNetPlugin_v6 = ibPlugin_v6;
      ncclNetPlugin_v5 = ibPlugin_v5;
      break;
  }

}

ncclResult_t pluginInit_v10(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  pluginLogFunction = logFunction;
  pluginSetup();
  INFO(NCCL_INIT|NCCL_NET, "P2P plugin v10 %s", ncclNetPlugin_v10.name);
  return ncclNetPlugin_v10.init(logFunction, profFunction);
}

ncclResult_t pluginInit_v9(ncclDebugLogger_t logFunction) {
  pluginLogFunction = logFunction;
  pluginSetup();
  INFO(NCCL_INIT|NCCL_NET, "P2P plugin v9 %s", ncclNetPlugin_v9.name);
  return ncclNetPlugin_v9.init(logFunction);
}

ncclResult_t pluginInit_v8(ncclDebugLogger_t logFunction) {
  pluginLogFunction = logFunction;
  pluginSetup();
  INFO(NCCL_INIT|NCCL_NET, "P2P plugin v8 %s", ncclNetPlugin_v8.name);
  return ncclNetPlugin_v8.init(logFunction);
}

ncclResult_t pluginInit_v7(ncclDebugLogger_t logFunction) {
  pluginLogFunction = logFunction;
  pluginSetup();
  INFO(NCCL_INIT|NCCL_NET, "P2P plugin %s", ncclNetPlugin_v7.name);
  return ncclNetPlugin_v7.init(logFunction);
}

ncclResult_t pluginInit_v6(ncclDebugLogger_t logFunction) {
  pluginLogFunction = logFunction;
  pluginSetup();
  INFO(NCCL_INIT|NCCL_NET, "P2P plugin %s", ncclNetPlugin_v6.name);
  return ncclNetPlugin_v6.init(logFunction);
}

ncclResult_t pluginInit_v5(ncclDebugLogger_t logFunction) {
  pluginLogFunction = logFunction;
  pluginSetup();
  INFO(NCCL_INIT|NCCL_NET, "P2P plugin %s", ncclNetPlugin_v5.name);
  return ncclNetPlugin_v5.init(logFunction);
}

// Detect whether GDR can work on a given NIC with the current CUDA device
// Returns :
// ncclSuccess : GDR works
// ncclSystemError : no module or module loaded but not supported by GPU
#define KNL_MODULE_LOADED(a) ((access(a, F_OK) == -1) ? 0 : 1)
static int ncclIbGdrModuleLoaded = 0; // 1 = true, 0 = false
static void ibGdrSupportInitOnce() {
  // Check for the nv_peer_mem module being loaded
  ncclIbGdrModuleLoaded = KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem/version") ||
                          KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem_nc/version") ||
                          KNL_MODULE_LOADED("/sys/module/nvidia_peermem/version");
}

ncclResult_t nccl_p2p_gdr_support()
{
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  pthread_once(&once, ibGdrSupportInitOnce);
  if (!ncclIbGdrModuleLoaded)
    return ncclSystemError;
  return ncclSuccess;
}

static __thread int ibDmaSupportInitDev; // which device to init, must be thread local
static void ibDmaBufSupportInitOnce(){
  ncclResult_t res;
  int dev_fail = 0;

  // This is a physical device, not a virtual one, so select from ibDevs
  ncclIbMergedDev* mergedDev = ncclIbMergedDevs + ibDmaSupportInitDev;
  ncclIbDev* ibDev = ncclIbDevs + mergedDev->vProps.devs[0];
  struct ibv_pd* pd;
  struct ibv_context* ctx = ibDev->context;
  NCCLCHECKGOTO(wrap_ibv_alloc_pd(&pd, ctx), res, failure);
  // Test kernel DMA-BUF support with a dummy call (fd=-1)
  (void)wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
  // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
  dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  NCCLCHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);
  // stop the search and goto failure
  if (dev_fail) goto failure;
  ibDev->dmaBufSupported = 1;
  return;
failure:
  ibDev->dmaBufSupported = -1;
  return;
}


struct oncewrap {
  pthread_once_t once;
};
static struct oncewrap onces[MAX_IB_DEVS];
// Detect whether DMA-BUF support is present in the kernel
// Returns :
// ncclSuccess : DMA-BUF support is available
// ncclSystemError : DMA-BUF is not supported by the kernel
ncclResult_t nccl_p2p_dmabuf_support(int dev) {
  // init the device only once
  ibDmaSupportInitDev = dev;
  pthread_once(&onces[dev].once, ibDmaBufSupportInitOnce);
  ncclIbMergedDev* mergedDev = ncclIbMergedDevs + ibDmaSupportInitDev;
  ncclIbDev* ibDev = ncclIbDevs + mergedDev->vProps.devs[0];
  int dmaBufSupported = ibDev->dmaBufSupported;
  if (dmaBufSupported == 1) return ncclSuccess;
  return ncclSystemError;
}

ncclResult_t ncclIbGetPhysProperties(int dev, ncclNetProperties_t* props) {
  struct ncclIbDev* ibDev = ncclIbDevs + dev;
  pthread_mutex_lock(&ibDev->lock);
  props->name = ibDev->devName;
  props->speed = ibDev->speed;
  props->pciPath = ibDev->pciPath;
  props->guid = ibDev->guid;
  props->ptrSupport   = NCCL_PTR_HOST;
  if (nccl_p2p_gdr_support() == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_CUDA; // GDR support via nv_peermem
    INFO(NCCL_NET,"NET/IB : GPU Direct RDMA (nvidia-peermem) enabled for HCA %d '%s", dev, ibDev->devName);
  }
  props->regIsGlobal = 1;
  props->forceFlush = 0;
  if (ibDev->capsProvider.mlx5.dataDirect) {
    props->forceFlush = 1;
  }
  if ((nccl_p2p_is_uct_plugin(p2p_plugin) || (p2p_plugin == NCCL_P2P_IB)) &&
      nccl_p2p_dmabuf_support(dev) == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_DMABUF; // GDR support via DMA-BUF
    INFO(NCCL_NET,"NET/IB : GPU Direct RDMA (DMABUF) enabled for HCA %d '%s", dev, ibDev->devName);
  }

  props->latency      = 0; // Not set
  props->port = ibDev->portNum + ibDev->realPort;
  props->maxComms = ibDev->maxQp;

  if (p2p_plugin == NCCL_P2P_IB || p2p_plugin == NCCL_P2P_UCX ||
      nccl_p2p_is_uct_plugin(p2p_plugin)) {
    props->maxRecvs = NCCL_NET_IB_MAX_RECVS;
  } else {
    props->maxRecvs = 1;
  }
  props->netDeviceType = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  pthread_mutex_unlock(&ibDev->lock);
  return ncclSuccess;
}

ncclResult_t nccl_p2p_ib_get_properties(ncclIbDev *devs, int ncclNMergedIbDevs, int dev, ncclNetProperties_t* props)
{
  if (dev >= ncclNMergedIbDevs) {
    WARN("NET/IB : Requested properties for vNic %d, only %d vNics have been created", dev, ncclNMergedIbDevs);
    return ncclInvalidUsage;
  }
  struct ncclIbMergedDev* mergedDev = ncclIbMergedDevs + dev;
  // Take the rest of the properties from an arbitrary sub-device (should be the same)
  NCCLCHECK(ncclIbGetPhysProperties(mergedDev->vProps.devs[0], props));
  props->name = mergedDev->devName;
  props->speed = mergedDev->speed;
  memcpy(&props->vProps, &mergedDev->vProps, sizeof(ncclNetVDeviceProps_t));
  return ncclSuccess;
}

ncclResult_t ncclIbStatsInit(struct ncclIbStats* stat) {
  __atomic_store_n(&stat->fatalErrorCount, 0, __ATOMIC_RELAXED);
  return ncclSuccess;
}

static void ncclIbStatsFatalError(struct ncclIbStats* stat){
  __atomic_fetch_add(&stat->fatalErrorCount, 1, __ATOMIC_RELAXED);
}
static void ncclIbQpFatalError(struct ibv_qp* qp) {
  ncclIbStatsFatalError((struct ncclIbStats*)qp->qp_context);
}
static void ncclIbCqFatalError(struct ibv_cq* cq) {
  ncclIbStatsFatalError((struct ncclIbStats*)cq->cq_context);
}
static void ncclIbDevFatalError(struct ncclIbDev* dev) {
  ncclIbStatsFatalError(&dev->stats);
}

static void* ncclIbAsyncThreadMain(void* args) {
  struct ncclIbDev* dev = (struct ncclIbDev*)args;
  while (1) {
    struct ibv_async_event event;
    if (ncclSuccess != wrap_ibv_get_async_event(dev->context, &event)) { break; }
    char *str;
    struct ibv_cq* cq = event.element.cq;    // only valid if CQ error
    struct ibv_qp* qp = event.element.qp;    // only valid if QP error
    struct ibv_srq* srq = event.element.srq; // only valid if SRQ error
    if (ncclSuccess != wrap_ibv_event_type_str(&str, event.event_type)) { break; }
    switch (event.event_type) {
    case IBV_EVENT_DEVICE_FATAL:
      // the above is device fatal error
      WARN("NET/IB : %s:%d async fatal event: %s", dev->devName, dev->portNum, str);
      ncclIbDevFatalError(dev);
      break;
    case IBV_EVENT_CQ_ERR:
      // the above is a CQ fatal error
      WARN("NET/IB : %s:%d async fatal event on CQ (%p): %s", dev->devName, dev->portNum, cq, str);
      ncclIbCqFatalError(cq);
      break;
    case IBV_EVENT_QP_FATAL:
    case IBV_EVENT_QP_REQ_ERR:
    case IBV_EVENT_QP_ACCESS_ERR:
      // the above are QP fatal errors
      WARN("NET/IB : %s:%d async fatal event on QP (%p): %s", dev->devName, dev->portNum, qp, str);
      ncclIbQpFatalError(qp);
      break;
    case IBV_EVENT_SRQ_ERR:
      // SRQ are not used in NCCL
      WARN("NET/IB : %s:%d async fatal event on SRQ, unused for now (%p): %s", dev->devName, dev->portNum, srq, str);
      break;
    case IBV_EVENT_PATH_MIG_ERR:
    case IBV_EVENT_PORT_ERR:
    case IBV_EVENT_PATH_MIG:
    case IBV_EVENT_PORT_ACTIVE:
    case IBV_EVENT_SQ_DRAINED:
    case IBV_EVENT_LID_CHANGE:
    case IBV_EVENT_PKEY_CHANGE:
    case IBV_EVENT_SM_CHANGE:
    case IBV_EVENT_QP_LAST_WQE_REACHED:
    case IBV_EVENT_CLIENT_REREGISTER:
    case IBV_EVENT_SRQ_LIMIT_REACHED:
      // the above are non-fatal
      WARN("NET/IB : %s:%d Got async error event: %s", dev->devName, dev->portNum, str);
      break;
    case IBV_EVENT_COMM_EST:
      break;
    default:
      WARN("NET/IB : %s:%d unknown event type (%d)", dev->devName, dev->portNum, event.event_type);
      break;
    }
    // acknowledgment needs to happen last to avoid user-after-free
    if (ncclSuccess != wrap_ibv_ack_async_event(&event)) { break; }
  }
  return NULL;
}

int devSharpCompare(const void *a, const void *b)
{
  const struct ncclIbDev *d1 = (const struct ncclIbDev *)a;
  const struct ncclIbDev *d2 = (const struct ncclIbDev *)b;

  if (d1->isSharpDev == d2->isSharpDev) { return 0; }
  else if (d1->isSharpDev > d2->isSharpDev) { return -1; }
  else { return 1; }
}


static bool ncclMlx5dvDmaBufCapable(struct ibv_context *context){
  ncclResult_t res;
  int dev_fail = 0;

  struct ibv_pd* pd;
  NCCLCHECKGOTO(wrap_ibv_alloc_pd(&pd, context), res, failure);
  // Test kernel DMA-BUF support with a dummy call (fd=-1)
  (void)wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
  // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
  (void)wrap_direct_mlx5dv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/, 0 /* mlx5 flags*/);
  // mlx5dv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
  dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  NCCLCHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);
  // stop the search and goto failure
  if (dev_fail) goto failure;
  return true;
failure:
  return false;
}

ncclResult_t ncclIbMakeVDeviceInternal(int* d, ncclNetVDeviceProps_t* props, int ncclNIbDevs, int *ncclNMergedIbDevs) {
  if ((ncclParamIbMergeNics() == 0) && props->ndevs > 1) {
    INFO(NCCL_NET, "NET/IB : Skipping makeVDevice, NCCL_IB_MERGE_NICS=0");
    return ncclInvalidUsage;
  }

  if (props->ndevs == 0) {
   WARN("NET/IB : Can't make virtual NIC with 0 devices");
   return ncclInvalidUsage;
  }

  if (*ncclNMergedIbDevs == MAX_IB_VDEVS) {
    WARN("NET/IB : Cannot allocate any more virtual devices (%d)", MAX_IB_VDEVS);
    return ncclInvalidUsage;
  }

  // Always count up number of merged devices
  ncclIbMergedDev* mDev = ncclIbMergedDevs + *ncclNMergedIbDevs;
  mDev->vProps.ndevs = 0;
  mDev->speed = 0;

  for (int i = 0; i < props->ndevs; i++) {
    ncclIbDev* dev = ncclIbDevs + props->devs[i];
    if (mDev->vProps.ndevs == NCCL_IB_MAX_DEVS_PER_NIC) return ncclInvalidUsage;
    mDev->vProps.devs[mDev->vProps.ndevs++] = props->devs[i];
    mDev->speed += dev->speed;
    // Each successive time, copy the name '+' new name
    if (mDev->vProps.ndevs > 1) {
      snprintf(mDev->devName + strlen(mDev->devName), sizeof(mDev->devName) - strlen(mDev->devName), "+%s", dev->devName);
    // First time, copy the plain name
    } else {
      strncpy(mDev->devName, dev->devName, MAXNAMESIZE);
     }
   }

  // Check link layers
  ncclIbDev* dev0 = ncclIbDevs + props->devs[0];
  for (int i = 1; i < props->ndevs; i++) {
    if (props->devs[i] >= ncclNIbDevs) {
      WARN("NET/IB : Cannot use physical device %d, max %d", props->devs[i], ncclNIbDevs);
      return ncclInvalidUsage;
    }
    ncclIbDev* dev = ncclIbDevs + props->devs[i];
    if (dev->link != dev0->link) {
      WARN("NET/IB : Attempted to merge incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
        props->devs[0], dev0->devName, dev0->portNum, NCCL_IB_LLSTR(dev0->link), props->devs[i], dev->devName, dev->portNum, NCCL_IB_LLSTR(dev->link)); 
      return ncclInvalidUsage;
    }
  }

  *d = *ncclNMergedIbDevs;
  (*ncclNMergedIbDevs)++;

  INFO(NCCL_NET, "NET/IB : Made virtual device [%d] name=%s speed=%d ndevs=%d", *d, mDev->devName, mDev->speed, mDev->vProps.ndevs);
  return ncclSuccess;
}

ncclResult_t nccl_p2p_ib_init(int *nDevs, int *nmDevs, ncclIbDev *ncclIbDevs, char *ncclIbIfName, union ncclSocketAddress *ncclIbIfAddr, pthread_t *ncclIbAsyncThread, ncclDebugLogger_t logFunction)
{
  ncclResult_t ret = ncclSuccess;
  int ncclNIbDevs = *nDevs;
  int ncclNMergedIbDevs = *nmDevs;
  pluginLogFunction = logFunction;
  if (ncclNIbDevs == -1) {
    for (int i=0; i< MAX_IB_DEVS; i++)
      onces[i].once = PTHREAD_ONCE_INIT;
    pthread_mutex_lock(&nccl_p2p_lock);
    wrap_ibv_fork_init();
    if (ncclNIbDevs == -1) {
      int nIpIfs = 0;
      ncclNIbDevs = 0;
      ncclNMergedIbDevs = 0;
      ncclNSharpDevs = 0;
      NCCLCHECK(ncclFindInterfaces(ncclIbIfName, ncclIbIfAddr, MAX_IF_NAME_SIZE, 1, &nIpIfs));
      if (nIpIfs != 1) {
        WARN("NET/IB : No IP interface found.");
        ret = ncclInternalError;
        goto fail;
      }

      // Detect IB cards
      int nIbDevs;
      struct ibv_device** devices;
      // Check if user defined which IB device:port to use
      const char* userIbEnv = ncclGetEnv("NCCL_IB_HCA");
      struct netIf userIfs[MAX_IB_DEVS];
      int searchNot = userIbEnv && userIbEnv[0] == '^';
      if (searchNot) userIbEnv++;
      int searchExact = userIbEnv && userIbEnv[0] == '=';
      if (searchExact) userIbEnv++;
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      if (ncclSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs)) { ret = ncclInternalError; goto fail; }
      for (int d=0; d<nIbDevs && ncclNIbDevs<MAX_IB_DEVS; d++) {
        struct ibv_context * context;
        if (ncclSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        enum ncclIbProvider ibProvider = IB_PROVIDER_NONE;
        char dataDirectDevicePath[PATH_MAX];
        int dataDirectSupported = 0;
        int skipNetDevForDataDirect = 0;
        if (wrap_mlx5dv_is_supported(devices[d])) {
          ibProvider = IB_PROVIDER_MLX5;
          snprintf(dataDirectDevicePath, PATH_MAX, "/sys");
          if((ncclMlx5dvDmaBufCapable(context)) && (wrap_mlx5dv_get_data_direct_sysfs_path(context, dataDirectDevicePath + 4, PATH_MAX - 4) == ncclSuccess)) {
            INFO(NCCL_INIT|NCCL_NET, "Data Direct DMA Interface is detected for device:%s", devices[d]->name);
            if(ncclParamIbDataDirect()) dataDirectSupported = 1;
            if(ncclParamIbDataDirect() == 2) skipNetDevForDataDirect = 1;
          }
        }
        int nPorts = 0;
        struct ibv_device_attr devAttr;
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context)) { ret = ncclInternalError; goto fail; }
          continue;
        }
        for (int port_num = 1; port_num <= devAttr.phys_port_cnt; port_num++) {
          for (int dataDirect = skipNetDevForDataDirect; dataDirect < 1 + dataDirectSupported; ++dataDirect) {
            struct ibv_port_attr portAttr;
            if (ncclSuccess != wrap_ibv_query_port(context, port_num, &portAttr)) {
              WARN("NET/IB : Unable to query port_num %d", port_num);
              continue;
            }
            if (portAttr.state != IBV_PORT_ACTIVE) continue;
            if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND
                && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;

            // check against user specified HCAs/ports
            if (! (matchIfList(devices[d]->name, port_num, userIfs, nUserIfs, searchExact) ^ searchNot)) {
              continue;
            }
            pthread_mutex_init(&ncclIbDevs[ncclNIbDevs].lock, NULL);
            ncclIbDevs[ncclNIbDevs].device = d;
            ncclIbDevs[ncclNIbDevs].ibProvider = ibProvider;
            ncclIbDevs[ncclNIbDevs].guid = devAttr.sys_image_guid;
            ncclIbDevs[ncclNIbDevs].portAttr = portAttr;
            ncclIbDevs[ncclNIbDevs].portNum = port_num;
            ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
            ncclIbDevs[ncclNIbDevs].speed = nccl_p2p_ib_speed(portAttr.active_speed_ex ? portAttr.active_speed_ex : portAttr.active_speed) * nccl_p2p_ib_width(portAttr.active_width);
            ncclIbDevs[ncclNIbDevs].context = context;
            ncclIbDevs[ncclNIbDevs].pdRefs = 0;
            ncclIbDevs[ncclNIbDevs].pd = NULL;
            if (!dataDirect) {
              strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
              NCCLCHECKGOTO(nccl_p2p_ib_pci_path(ncclIbDevs, ncclNIbDevs, ncclIbDevs[ncclNIbDevs].devName, &ncclIbDevs[ncclNIbDevs].pciPath, &ncclIbDevs[ncclNIbDevs].realPort), ret, fail);
            } else {
              snprintf(ncclIbDevs[ncclNIbDevs].devName, MAXNAMESIZE, "%s_dma", devices[d]->name);
              ncclIbDevs[ncclNIbDevs].pciPath = malloc(PATH_MAX);
              strncpy(ncclIbDevs[ncclNIbDevs].pciPath, dataDirectDevicePath, PATH_MAX);
              ncclIbDevs[ncclNIbDevs].capsProvider.mlx5.dataDirect = 1;
            }
            ncclIbDevs[ncclNIbDevs].maxQp = devAttr.max_qp;
            ncclIbDevs[ncclNIbDevs].mrCache.capacity = 0;
            ncclIbDevs[ncclNIbDevs].mrCache.population = 0;
            ncclIbDevs[ncclNIbDevs].mrCache.slots = NULL;
            NCCLCHECK(ncclIbStatsInit(&ncclIbDevs[ncclNIbDevs].stats));

          // Enable ADAPTIVE_ROUTING by default on IB networks
            // But allow it to be overloaded by an env parameter
            ncclIbDevs[ncclNIbDevs].ar = (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
            if (ncclParamIbAdaptiveRouting() != -2) ncclIbDevs[ncclNIbDevs].ar = ncclParamIbAdaptiveRouting();

            ncclIbDevs[ncclNIbDevs].isSharpDev = 0;
            if (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND)
            {
              ncclIbDevs[ncclNIbDevs].isSharpDev = 1;
              ncclIbDevs[ncclNIbDevs].maxQp = ncclParamSharpMaxComms();
              ncclNSharpDevs++;
            }
            TRACE(NCCL_NET,"NET/IB: [%d] %s:%s:%d/%s provider=%s speed=%d context=%p pciPath=%s ar=%d", d, devices[d]->name, devices[d]->dev_name, ncclIbDevs[ncclNIbDevs].portNum,
              NCCL_IB_LLSTR(portAttr.link_layer), ibProviderName[ncclIbDevs[ncclNIbDevs].ibProvider], ncclIbDevs[ncclNIbDevs].speed, context, ncclIbDevs[ncclNIbDevs].pciPath, ncclIbDevs[ncclNIbDevs].ar);
            if (ncclIbAsyncThread != NULL) {
              PTHREADCHECKGOTO(pthread_create(ncclIbAsyncThread, NULL, ncclIbAsyncThreadMain, ncclIbDevs + ncclNIbDevs), "pthread_create", ret, fail);
              ncclSetThreadName(*ncclIbAsyncThread, "NCCL IbAsync %2d", ncclNIbDevs);
              PTHREADCHECKGOTO(pthread_detach(*ncclIbAsyncThread), "pthread_detach", ret, fail); // will not be pthread_join()'d
            }

            // Add this plain physical device to the list of virtual devices
            int vDev;
            ncclNetVDeviceProps_t vProps = {0};
            vProps.ndevs = 1;
            vProps.devs[0] = ncclNIbDevs;
            NCCLCHECK(ncclIbMakeVDeviceInternal(&vDev, &vProps, ncclNIbDevs, &ncclNMergedIbDevs));

            ncclNIbDevs++;
            nPorts++;
          }
        }
        if (nPorts == 0 && ncclSuccess != wrap_ibv_close_device(context))  { ret = ncclInternalError; goto fail; }
      }
      
      if (nIbDevs && (ncclSuccess != wrap_ibv_free_device_list(devices))) { ret = ncclInternalError; goto fail; };
    }
    if (ncclNIbDevs == 0) {
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : No device found.");
    }

    // Print out all net devices to the user (in the same format as before)
    char line[2048];
    line[0] = '\0';
    // Determine whether RELAXED_ORDERING is enabled and possible
    ncclIbRelaxedOrderingEnabled = ncclIbRelaxedOrderingCapable();
    for (int d = 0; d < ncclNIbDevs; d++) {
#ifdef HAVE_SHARP_PLUGIN
            snprintf(line+strlen(line), sizeof(line)-strlen(line), " [%d]%s:%d/%s%s", d, ncclIbDevs[d].devName,
              ncclIbDevs[d].portNum, NCCL_IB_LLSTR(ncclIbDevs[d].link),
              ncclIbDevs[d].isSharpDev ? "/SHARP" : "");
#else
      snprintf(line+strlen(line), sizeof(line)-strlen(line), " [%d]%s:%d/%s", d, ncclIbDevs[d].devName,
        ncclIbDevs[d].portNum, NCCL_IB_LLSTR(ncclIbDevs[d].link));
#endif
    }
    char addrline[SOCKET_NAME_MAXLEN+1];
    INFO(NCCL_INIT|NCCL_NET, "NET/IB : Using%s %s; OOB %s:%s", line, ncclIbRelaxedOrderingEnabled ? "[RO]" : "",
      ncclIbIfName, ncclSocketToString(ncclIbIfAddr, addrline, 1));
    *nDevs = ncclNIbDevs;
    *nmDevs = ncclNMergedIbDevs;
    pthread_mutex_unlock(&nccl_p2p_lock);
  }
exit:
  return ret;
fail:
  pthread_mutex_unlock(&nccl_p2p_lock);
  goto exit;
}

// Returns 0 if this is the path of two VFs of the same physical device
static int ncclIbMatchVfPath(char* path1, char* path2) {
  // Merge multi-port NICs into the same PCI device
  if (ncclParamIbMergeVfs()) {
    return strncmp(path1, path2, strlen(path1)-4) == 0;
  } else {
    return strncmp(path1, path2, strlen(path1)-1) == 0;
  }
}

ncclResult_t nccl_p2p_ib_pci_path(ncclIbDev *devs, int num_devs, char* dev_name, char** path, int* real_port)
{
  char device_path[PATH_MAX];
  snprintf(device_path, PATH_MAX, "/sys/class/infiniband/%s/device", dev_name);
  char* p = realpath(device_path, NULL);
  if (p == NULL) {
    WARN("Could not find real path of %s", *device_path);
  } else {
    // Merge multi-port NICs into the same PCI device
    p[strlen(p)-1] = '0';
    // Also merge virtual functions (VF) into the same device
    if (ncclParamIbMergeVfs()) p[strlen(p)-3] = p[strlen(p)-4] = '0';
    // Keep the real port aside (the ibv port is always 1 on recent cards)
    *real_port = 0;
    for (int d=0; d<num_devs; d++) {
      if (ncclIbMatchVfPath(p, ncclIbDevs[d].pciPath)) (*real_port)++;
    }
  }
  *path = p;
  return ncclSuccess;
}

static int ibv_widths[] = { 1, 4, 8, 12, 2};
static int ibv_speeds[] = {
  2500,  /* SDR */
  5000,  /* DDR */
  10000, /* QDR */
  10000, /* QDR */
  14000, /* FDR */
  25000, /* EDR */
  50000, /* HDR */
  100000, /* NDR */
  200000  /* XDR */
};


static int first_bit_set(int val, int max) {
  int i = 0;
  while (i<max && ((val & (1<<i)) == 0)) i++;
  return i;
}

int nccl_p2p_ib_width(int width)
{
  return ibv_widths[first_bit_set(width, sizeof(ibv_widths)/sizeof(int)-1)];
}

int nccl_p2p_ib_speed(int speed)
{
  return ibv_speeds[first_bit_set(speed, sizeof(ibv_speeds)/sizeof(int)-1)];
}

nccl_p2p_plugin_t nccl_p2p_get_plugin_type()
{
  return p2p_plugin;
}

struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
struct ncclIbDev userIbDevs[MAX_IB_DEVS];
struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_VDEVS];
