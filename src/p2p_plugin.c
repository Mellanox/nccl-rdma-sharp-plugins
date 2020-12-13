/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 * Copyright (C) 2019-2020, Mellanox Technologies Ltd. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <strings.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/time.h>

#include "nccl.h"
#include "nccl_net.h"
#include "debug.h"
#include "p2p_plugin.h"

#ifdef HAVE_UCX_PLUGIN
extern ncclNet_t ucxPlugin;
extern ncclNet_t ucxRmaPlugin;
#endif

extern ncclNet_t ibPlugin;
pthread_mutex_t nccl_p2p_lock = PTHREAD_MUTEX_INITIALIZER;

ncclDebugLogger_t pluginLogFunction;

#ifdef HAVE_SHARP_PLUGIN
extern int ncclNSharpDevs;
#else
/* In case sharp plugin is not there just define this variable locally to make code cleaner */
int ncclNSharpDevs;
#endif
NCCL_PARAM(SharpMaxComms, "SHARP_MAX_COMMS", 1);

ncclResult_t pluginInit(ncclDebugLogger_t logFunction);

ncclNet_t NCCL_PLUGIN_SYMBOL = {
  "NCCL RDMA Plugin",
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

static nccl_p2p_plugin_t p2p_plugin = NCCL_P2P_LAST;

ncclResult_t pluginInit(ncclDebugLogger_t logFunction)
{
  pluginLogFunction = logFunction;
  p2p_plugin = NCCL_P2P_IB;
  const char *plugin_path = get_plugin_lib_path();
  if (plugin_path != NULL) {
    INFO(NCCL_INIT|NCCL_NET, "Plugin Path : %s", plugin_path);;
  }

  NCCL_PLUGIN_SYMBOL = ibPlugin;
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
      NCCL_PLUGIN_SYMBOL = ibPlugin;
      break;
#ifdef HAVE_UCX_PLUGIN
    case NCCL_P2P_UCX:
      NCCL_PLUGIN_SYMBOL = ucxPlugin;
      break;
    case NCCL_P2P_UCX_RMA:
      NCCL_PLUGIN_SYMBOL = ucxRmaPlugin;
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
  props->speed        = devs[dev].speed;
  props->port         = devs[dev].port + devs[dev].realPort;
  props->maxComms     = devs[dev].maxQp;

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

ncclResult_t nccl_p2p_ib_init(int *num_devs, nccl_ib_dev_t *ncclIbDevs, char *ncclIbIfName, union socketAddress *ncclIbIfAddr, pthread_t *ncclIbAsyncThread, ncclDebugLogger_t logFunction)
{
  int ncclNIbDevs = *num_devs;

  pluginLogFunction = logFunction;
  if (ncclNIbDevs == -1) {
    pthread_mutex_lock(&nccl_p2p_lock);
    wrap_ibv_fork_init();
    if (ncclNIbDevs == -1) {
      ncclNIbDevs = 0;
      ncclNSharpDevs = 0;
      if (findInterfaces(ncclIbIfName, ncclIbIfAddr, MAX_IF_NAME_SIZE, 1) != 1) {
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
        int found = 0;
        struct ibv_device_attr devAttr;
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context)) { return ncclInternalError; }
          continue;
        }
        for (int port = 1; port <= devAttr.phys_port_cnt; port++) {
          struct ibv_port_attr portAttr;
          long vendorId, devId;
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
          ncclIbDevs[ncclNIbDevs].device = d;
          ncclIbDevs[ncclNIbDevs].guid = devAttr.sys_image_guid;
          ncclIbDevs[ncclNIbDevs].port = port;
          ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
          ncclIbDevs[ncclNIbDevs].speed = nccl_p2p_ib_speed(portAttr.active_speed) * nccl_p2p_ib_width(portAttr.active_width);
          ncclIbDevs[ncclNIbDevs].context = context;
          strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
          NCCLCHECK(nccl_p2p_ib_pci_path(ncclIbDevs, ncclNIbDevs, ncclIbDevs[ncclNIbDevs].devName, &ncclIbDevs[ncclNIbDevs].pciPath, &ncclIbDevs[ncclNIbDevs].realPort));
          ncclIbDevs[ncclNIbDevs].maxQp = devAttr.max_qp;
          readFileNumber(&vendorId, IB_DEVICE_SYSFS_FMT, devices[d]->name, "vendor");
          readFileNumber(&devId, IB_DEVICE_SYSFS_FMT, devices[d]->name, "device");
          ncclIbDevs[ncclNIbDevs].isSharpDev = 0;
          if ((portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) &&
              (vendorId == 0x15b3) &&           // Mellanox vendor
              (devId == 4123 || devId == 4124)) //ConnectX-6
          {
            ncclIbDevs[ncclNIbDevs].isSharpDev = 1;
            ncclIbDevs[ncclNIbDevs].maxQp = ncclParamSharpMaxComms();
            ncclNSharpDevs++;
          }
          ncclNIbDevs++;
          found++;

          if (ncclIbAsyncThread != NULL) {
            pthread_create(ncclIbAsyncThread, NULL, ncclIbAsyncThreadMain, context);
          }
        }
        if (found == 0 && ncclSuccess != wrap_ibv_close_device(context)) { return ncclInternalError; }
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
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : Using%s ; OOB %s:%s", line, ncclIbIfName, socketToString(&ncclIbIfAddr->sa, addrline));
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
    // And keep the real port aside (the ibv port is always 1 on recent cards)
    *real_port = 0;
    for (int d=0; d<num_devs; d++) {
      if (strcmp(p, devs[d].pciPath) == 0) (*real_port)++;
    }
  }
  *path = p;
  return ncclSuccess;
}

static int ibv_widths[] = { 1, 4, 8, 12 };
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

