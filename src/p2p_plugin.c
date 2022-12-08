/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
extern ncclNet_v6_t ucxPlugin_v6;
extern ncclNet_v5_t ucxPlugin_v5;
extern ncclNet_v4_t ucxPlugin_v4;
extern ncclNet_v6_t ucxRmaPlugin_v6;
extern ncclNet_v5_t ucxRmaPlugin_v5;
extern ncclNet_v4_t ucxRmaPlugin_v4;
#endif

extern ncclNet_v6_t ibPlugin_v6;
extern ncclNet_v5_t ibPlugin_v5;
extern ncclNet_v4_t ibPlugin_v4;
pthread_mutex_t nccl_p2p_lock = PTHREAD_MUTEX_INITIALIZER;

ncclDebugLogger_t pluginLogFunction;

#ifdef HAVE_SHARP_PLUGIN
extern int ncclNSharpDevs;
#else
/* In case sharp plugin is not there just define this variable locally to make code cleaner */
int ncclNSharpDevs;
#endif
extern int ncclIbRelaxedOrderingEnabled;
NCCL_PARAM(SharpMaxComms, "SHARP_MAX_COMMS", 1);

ncclResult_t pluginInit(ncclDebugLogger_t logFunction);

ncclNet_v6_t ncclNetPlugin_v6 = {
  "NCCL RDMA Plugin v6",
  pluginInit,
};


ncclNet_v5_t ncclNetPlugin_v5 = {
  "NCCL RDMA Plugin v5",
  pluginInit,
};


ncclNet_v4_t ncclNetPlugin_v4 = {
  "NCCL RDMA Plugin v4",
  pluginInit,
};

static nccl_p2p_plugin_t p2p_plugin = NCCL_P2P_LAST;

ncclResult_t pluginInit(ncclDebugLogger_t logFunction)
{
  pluginLogFunction = logFunction;

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
#endif
    else {
      WARN("Invalid value %s for NCCL_PLUGIN_P2P, using default", p2p_layer);
    }
  }
  switch (p2p_plugin) {
    case NCCL_P2P_IB:
      ncclNetPlugin_v6 = ibPlugin_v6;
      ncclNetPlugin_v5 = ibPlugin_v5;
      ncclNetPlugin_v4 = ibPlugin_v4;
      break;
#ifdef HAVE_UCX_PLUGIN
    case NCCL_P2P_UCX:
      ncclNetPlugin_v6 = ucxPlugin_v6;
      ncclNetPlugin_v5 = ucxPlugin_v5;
      ncclNetPlugin_v4 = ucxPlugin_v4;
      break;
    case NCCL_P2P_UCX_RMA:
      ncclNetPlugin_v6 = ucxRmaPlugin_v6;
      ncclNetPlugin_v5 = ucxRmaPlugin_v5;
      ncclNetPlugin_v4 = ucxRmaPlugin_v4;
      break;
#endif
  }

  INFO(NCCL_INIT|NCCL_NET, "P2P plugin %s", NCCL_PLUGIN_SYMBOL.name);
  return NCCL_PLUGIN_SYMBOL.init(logFunction);
}

ncclResult_t nccl_p2p_gdr_support(int dev)
{
  static int module_loaded = -1;

  if (module_loaded == -1) {
    module_loaded = (access("/sys/kernel/mm/memory_peers/nv_mem/version", F_OK) == -1) ? 0 : 1;
  }

  if (module_loaded == 0) {
      return ncclSystemError;
  }

  return ncclSuccess;
}

// Detect whether DMA-BUF support is present in the kernel
// Returns :
// ncclSuccess : DMA-BUF support is available
// ncclSystemError : DMA-BUF is not supported by the kernel
ncclResult_t nccl_p2p_dmabuf_support(int dev) {
  static int dmaBufSupported = -1;
  if (dmaBufSupported == -1) {
    ncclResult_t res;
    struct ibv_pd* pd;
    struct ibv_context* ctx;
    ctx = ncclIbDevs[dev].context;
    NCCLCHECKGOTO(wrap_ibv_alloc_pd(&pd, ctx), res, failure);
    // Test kernel DMA-BUF support with a dummy call (fd=-1)
    (void) wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL/*offset*/, 0ULL/*len*/, 0ULL/*iova*/, -1/*fd*/, 0/*flags*/);
    // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
    dmaBufSupported = (errno != EOPNOTSUPP && errno != EPROTONOSUPPORT) ? 1 : 0;
    NCCLCHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);
  }
  if (dmaBufSupported == 0) return ncclSystemError;
  return ncclSuccess;
failure:
  dmaBufSupported = 0;
  return ncclSystemError;
}


ncclResult_t nccl_p2p_ib_get_properties(nccl_ib_dev_t *devs, int dev, ncclNetProperties_t* props)
{
  props->name         = devs[dev].devName;
  props->pciPath      = devs[dev].pciPath;
  props->guid         = devs[dev].guid;
  props->ptrSupport   = NCCL_PTR_HOST;
  if (nccl_p2p_gdr_support(dev) != ncclSuccess) {
    INFO(NCCL_NET,"NET/IB : GPU Direct RDMA Disabled for HCA %d '%s' (no module)", dev, devs[dev].devName);
  } else {
    props->ptrSupport |= NCCL_PTR_CUDA;
  }
  if (nccl_p2p_gdr_support(dev) == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_CUDA; // GDR support via nv_peermem
  }
  if (p2p_plugin == NCCL_P2P_IB && nccl_p2p_dmabuf_support(dev) == ncclSuccess) {
    props->ptrSupport |= NCCL_PTR_DMABUF; // GDR support via DMA-BUF
  }

  props->speed        = devs[dev].speed;
  props->latency      = 0; // Not set
  props->port         = devs[dev].port + devs[dev].realPort;
  props->maxComms     = devs[dev].maxQp;
  props->maxRecvs     = (p2p_plugin == NCCL_P2P_IB) ? NCCL_NET_IB_MAX_RECVS : 1;

  return ncclSuccess;
}

static void* ncclIbAsyncThreadMain(void* args) {
  struct ibv_context* context = (struct ibv_context*)args;
  while (1) {
    struct ibv_async_event event;
    if (ncclSuccess != wrap_ibv_get_async_event(context, &event)) { break; }
    char *str;
    if (ncclSuccess != wrap_ibv_event_type_str(&str, event.event_type)) { break; }
    if (event.event_type != IBV_EVENT_COMM_EST)
      WARN("NET/IB : Got async event : %s", str);
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

ncclResult_t nccl_p2p_ib_init(int *num_devs, nccl_ib_dev_t *ncclIbDevs, char *ncclIbIfName, union ncclSocketAddress *ncclIbIfAddr, pthread_t *ncclIbAsyncThread, ncclDebugLogger_t logFunction)
{
  int ncclNIbDevs = *num_devs;

  pluginLogFunction = logFunction;
  if (ncclNIbDevs == -1) {
    pthread_mutex_lock(&nccl_p2p_lock);
    wrap_ibv_fork_init();
    if (ncclNIbDevs == -1) {
      ncclNIbDevs = 0;
      ncclNSharpDevs = 0;
      if (ncclFindInterfaces(ncclIbIfName, ncclIbIfAddr, MAX_IF_NAME_SIZE, 1) != 1) {
        WARN("NET/IB : No IP interface found.");
        return ncclInternalError;
      }

      // Detect IB cards
      int nIbDevs;
      struct ibv_device** devices;
      // Check if user defined which IB device:port to use
      char* userIbEnv = getenv("NCCL_IB_HCA");
      struct netIf userIfs[MAX_IB_DEVS];
      int searchNot = userIbEnv && userIbEnv[0] == '^';
      if (searchNot) userIbEnv++;
      int searchExact = userIbEnv && userIbEnv[0] == '=';
      if (searchExact) userIbEnv++;
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);

      if (ncclSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs)) return ncclInternalError;

      for (int d=0; d<nIbDevs; d++) {
        struct ibv_context * context;
        if (ncclSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        int nPorts = 0;
        struct ibv_device_attr devAttr;
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context)) { return ncclInternalError; }
          continue;
        }
        for (int port = 1; port <= devAttr.phys_port_cnt; port++) {
          struct ibv_port_attr portAttr;
          if (ncclSuccess != wrap_ibv_query_port(context, port, &portAttr)) {
            WARN("NET/IB : Unable to query port %d", port);
            continue;
          }
          if (portAttr.state != IBV_PORT_ACTIVE) continue;
          if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND
              && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;

          // check against user specified HCAs/ports
          if (! (matchIfList(devices[d]->name, port, userIfs, nUserIfs, searchExact) ^ searchNot)) {
            continue;
          }
          TRACE(NCCL_INIT|NCCL_NET,"NET/IB: [%d] %s:%d/%s ", d, devices[d]->name, port,
              portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");
          pthread_mutex_init(&ncclIbDevs[ncclNIbDevs].lock, NULL);
          ncclIbDevs[ncclNIbDevs].device = d;
          ncclIbDevs[ncclNIbDevs].guid = devAttr.sys_image_guid;
          ncclIbDevs[ncclNIbDevs].port = port;
          ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
          ncclIbDevs[ncclNIbDevs].speed = nccl_p2p_ib_speed(portAttr.active_speed) * nccl_p2p_ib_width(portAttr.active_width);
          ncclIbDevs[ncclNIbDevs].context = context;
          ncclIbDevs[ncclNIbDevs].pdRefs = 0;
          ncclIbDevs[ncclNIbDevs].pd = NULL;
          strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
          NCCLCHECK(nccl_p2p_ib_pci_path(ncclIbDevs, ncclNIbDevs, ncclIbDevs[ncclNIbDevs].devName, &ncclIbDevs[ncclNIbDevs].pciPath, &ncclIbDevs[ncclNIbDevs].realPort));
          ncclIbDevs[ncclNIbDevs].maxQp = devAttr.max_qp;
          ncclIbDevs[ncclNIbDevs].mrCache.capacity = 0;
          ncclIbDevs[ncclNIbDevs].mrCache.population = 0;
          ncclIbDevs[ncclNIbDevs].mrCache.slots = NULL;
          ncclIbDevs[ncclNIbDevs].isSharpDev = 0;
          if (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND)
          {
            ncclIbDevs[ncclNIbDevs].isSharpDev = 1;
            ncclIbDevs[ncclNIbDevs].maxQp = ncclParamSharpMaxComms();
            ncclNSharpDevs++;
          }

          if (ncclIbAsyncThread != NULL) {
            pthread_create(ncclIbAsyncThread, NULL, ncclIbAsyncThreadMain, context);
            ncclSetThreadName(*ncclIbAsyncThread, "NCCL IbAsync %2d", ncclNIbDevs);
            pthread_detach(*ncclIbAsyncThread); // will not be pthread_join()'d
          }
          ncclNIbDevs++;
          nPorts++;
        }
        if (nPorts == 0 && ncclSuccess != wrap_ibv_close_device(context)) { return ncclInternalError; }
      }
      if (nIbDevs && (ncclSuccess != wrap_ibv_free_device_list(devices))) { return ncclInternalError; };
    }
    if (ncclNIbDevs == 0) {
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : No device found.");
    } else {
      // sort devices on sharp capable
      if (ncclNSharpDevs && (ncclNSharpDevs != ncclNIbDevs)) {
        qsort(ncclIbDevs, ncclNIbDevs, sizeof(struct ncclIbDev), devSharpCompare);
      }

      char line[1024];
      line[0] = '\0';
      // Determine whether RELAXED_ORDERING is enabled and possible
      ncclIbRelaxedOrderingEnabled = ncclIbRelaxedOrderingCapable();
      for (int d=0; d<ncclNIbDevs; d++) {
#ifdef HAVE_SHARP_PLUGIN
        snprintf(line+strlen(line), 1023-strlen(line), " [%d]%s:%d/%s%s", d, ncclIbDevs[d].devName,
            ncclIbDevs[d].port, ncclIbDevs[d].link == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE",
            ncclIbDevs[d].isSharpDev ? "/SHARP" : "");
#else
        snprintf(line+strlen(line), 1023-strlen(line), " [%d]%s:%d/%s", d, ncclIbDevs[d].devName,
            ncclIbDevs[d].port, ncclIbDevs[d].link == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");
#endif
      }
      line[1023] = '\0';
      char addrline[SOCKET_NAME_MAXLEN+1];
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : Using%s %s; OOB %s:%s", line, ncclIbRelaxedOrderingEnabled ? "[RO]" : "",
           ncclIbIfName, ncclSocketToString(ncclIbIfAddr, addrline, 1));
    }
    *num_devs = ncclNIbDevs;
    pthread_mutex_unlock(&nccl_p2p_lock);
  }
  return ncclSuccess;

}

ncclResult_t nccl_p2p_ib_pci_path(nccl_ib_dev_t *devs, int num_devs, char* dev_name, char** path, int* real_port)
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
    p[strlen(p)-3] = '0';
    // And keep the real port aside (the ibv port is always 1 on recent cards)
    *real_port = 0;
    for (int d=0; d<num_devs; d++) {
      if (strcmp(p, devs[d].pciPath) == 0) (*real_port)++;
    }
  }
  *path = p;
  return ncclSuccess;
}

static int ibv_widths[] = { 1, 4, 8, 12, 2};
static int ibv_speeds[] = { 2500, 5000, 10000, 10000, 14000, 25000, 50000 };

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

