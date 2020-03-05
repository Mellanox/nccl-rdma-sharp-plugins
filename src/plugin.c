/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>

#include "nccl.h"
#include "nccl_net.h"
#include "core.h"
#include "socket.h"
#include "utils.h"
#include "param.h"

#include <assert.h>
#include "ibvwrap.h"

#define USE_RDMA_WRITE 1
#define USE_RDMA_SEND_INLINE 0
#define MAXNAMESIZE 64
 #define IB_DEVICE_SYSFS_FMT "/sys/class/infiniband/%s/device/%s"
static char ncclIbIfName[MAX_IF_NAME_SIZE];
static union socketAddress ncclIbIfAddr;
static int ncclNIbDevs = -1;
static int ncclNSharpDevs = -1;
struct ncclIbDev {
  int device;
  uint64_t guid;
  uint8_t port;
  uint8_t link;
  uint8_t isSharpDev;
  int speed;
  struct ibv_context* context;
  char devName[MAXNAMESIZE];
  char* pciPath;
  int realPort;
  int maxQp;
};

#define MAX_IB_PORT 15
struct userIbDev {
  char devName[MAXNAMESIZE];
  uint16_t port_en;
};

#define MAX_IB_DEVS 16
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
struct userIbDev userIbDevs[MAX_IB_DEVS];
pthread_mutex_t ncclIbLock = PTHREAD_MUTEX_INITIALIZER;

NCCL_PARAM(IbGidIndex, "IB_GID_INDEX", 0);
NCCL_PARAM(IbTimeout, "IB_TIMEOUT", 14);
NCCL_PARAM(IbRetryCnt, "IB_RETRY_CNT", 7);
NCCL_PARAM(IbSl, "IB_SL", 0);
NCCL_PARAM(IbTc, "IB_TC", 0);
NCCL_PARAM(IbPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 0);
NCCL_PARAM(SharpMaxComms, "SHARP_MAX_COMMS", 1);

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked DONTFORK
// and if they are shared, that could cause a crash in a child process
static ncclResult_t ncclIbMalloc(void** ptr, size_t size) {
  size_t page_size = sysconf(_SC_PAGESIZE);
  void* p;
  int size_aligned = ROUNDUP(size, page_size);
  int ret = posix_memalign(&p, page_size, size_aligned);
  if (ret != 0) return ncclSystemError;
  memset(p, 0, size);
  *ptr = p;
  return ncclSuccess;
}

pthread_t ncclIbAsyncThread;
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

NCCL_PARAM(IbDisable, "IBEXT_DISABLE", 0);

static ncclResult_t ncclIbGetPciPath(char* devName, char** path, int* realPort) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
  char* p = realpath(devicePath, NULL);
  if (p == NULL) {
    WARN("Could not find real path of %s", *devicePath);
  } else {
    // Merge multi-port NICs into the same PCI device
    p[strlen(p)-1] = '0';
    // And keep the real port aside (the ibv port is always 1 on recent cards)
    *realPort = 0;
    for (int d=0; d<ncclNIbDevs; d++) {
      if (strcmp(p, ncclIbDevs[d].pciPath) == 0) (*realPort)++;
    }
  }
  *path = p;
  return ncclSuccess;
}

static int ibvWidths[] = { 1, 4, 8, 12 };
static int ibvSpeeds[] = { 2500, 5000, 10000, 10000, 14000, 25000, 50000 };
static int firstBitSet(int val, int max) {
  int i = 0;
  while (i<max && ((val & (1<<i)) == 0)) i++;
  return i;
}
static int ncclIbWidth(int width) {
  return ibvWidths[firstBitSet(width, sizeof(ibvWidths)/sizeof(int)-1)];
}
static int ncclIbSpeed(int speed) {
  return ibvSpeeds[firstBitSet(speed, sizeof(ibvSpeeds)/sizeof(int)-1)];
}

ncclDebugLogger_t pluginLogFunction;

int devCompare(const void *a, const void *b) {
  const struct ncclIbDev *d1 = (const struct ncclIbDev *)a;
  const struct ncclIbDev *d2 = (const struct ncclIbDev *)b;

  if (d1->isSharpDev == d2->isSharpDev) { return 0; }
  else if (d1->isSharpDev > d2->isSharpDev) { return -1; }
  else { return 1; }
}

ncclResult_t ncclIbInit(ncclDebugLogger_t logFunction) {
  struct timeval tval;
  gettimeofday(&tval, NULL);
  srand((int) tval.tv_usec);

  pluginLogFunction = logFunction;

  if (ncclParamIbDisable()) return ncclInternalError;

  if (ncclNIbDevs == -1) {
    pthread_mutex_lock(&ncclIbLock);
    wrap_ibv_fork_init();

    if (ncclParamIbPciRelaxedOrdering() && !IBV_ACCESS_RELAXED_ORDERING) {
      WARN("NET/IB: PCI relaxed order memory access requested but not supported");
    }

    if (ncclNIbDevs == -1) {
      ncclNIbDevs = 0;
      ncclNSharpDevs = 0;
      if (findInterfaces(ncclIbIfName, &ncclIbIfAddr, MAX_IF_NAME_SIZE, 1) != 1) {
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
          if (! (matchIfList(devices[d]->name, port, userIfs, nUserIfs) ^ searchNot)) {
            continue;
          }
          TRACE(NCCL_INIT|NCCL_NET,"NET/IB: [%d] %s:%d/%s ", d, devices[d]->name, port,
              portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");
          ncclIbDevs[ncclNIbDevs].device = d;
          ncclIbDevs[ncclNIbDevs].guid = devAttr.sys_image_guid;
          ncclIbDevs[ncclNIbDevs].port = port;
          ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
	  ncclIbDevs[ncclNIbDevs].speed = ncclIbSpeed(portAttr.active_speed) * ncclIbWidth(portAttr.active_width);
          ncclIbDevs[ncclNIbDevs].context = context;
          strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
	  NCCLCHECK(ncclIbGetPciPath(ncclIbDevs[ncclNIbDevs].devName, &ncclIbDevs[ncclNIbDevs].pciPath, &ncclIbDevs[ncclNIbDevs].realPort));
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
          pthread_create(&ncclIbAsyncThread, NULL, ncclIbAsyncThreadMain, context);
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
        qsort(ncclIbDevs, ncclNIbDevs, sizeof(struct ncclIbDev), devCompare);
      }

      char line[1024];
      line[0] = '\0';
      for (int d=0; d<ncclNIbDevs; d++) {
        snprintf(line+strlen(line), 1023-strlen(line), " [%d]%s:%d/%s%s", d, ncclIbDevs[d].devName,
            ncclIbDevs[d].port, ncclIbDevs[d].link == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE",
            ncclIbDevs[d].isSharpDev ? "/SHARP" : "");
      }
      line[1023] = '\0';
      char addrline[1024];
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : Using%s ; OOB %s:%s", line, ncclIbIfName, socketToString(&ncclIbIfAddr.sa, addrline));
    }
    pthread_mutex_unlock(&ncclIbLock);
  }
  return ncclSuccess;
}

ncclResult_t ncclIbDevices(int* ndev) {
  *ndev = ncclNIbDevs;
  return ncclSuccess;
}

ncclResult_t ncclIbSharpDevices(int* ndev) {
  *ndev = ncclNSharpDevs;
  return ncclSuccess;
}

// Detect whether GDR can work on a given NIC with the current CUDA device
// Returns :
// ncclSuccess : GDR works
// ncclSystemError : no module or module loaded but not supported by GPU
ncclResult_t ncclIbGdrSupport(int ibDev) {
  static int moduleLoaded = -1;
  if (moduleLoaded == -1) {
    moduleLoaded = (access("/sys/kernel/mm/memory_peers/nv_mem/version", F_OK) == -1) ? 0 : 1;
  }
  if (moduleLoaded == 0) return ncclSystemError;
  return ncclSuccess;
}

ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props)
{
  props->name = ncclIbDevs[dev].devName;
  props->pciPath = ncclIbDevs[dev].pciPath;
  props->guid = ncclIbDevs[dev].guid;
  props->ptrSupport = NCCL_PTR_HOST;
  if (ncclIbGdrSupport(dev) != ncclSuccess) {
    INFO(NCCL_NET,"NET/IB : GPU Direct RDMA Disabled for HCA %d '%s' (no module)", dev, ncclIbDevs[dev].devName);
  } else {
    props->ptrSupport |= NCCL_PTR_CUDA;
  }
  props->speed = ncclIbDevs[dev].speed;
  props->port = ncclIbDevs[dev].port + ncclIbDevs[dev].realPort;
  props->maxComms = ncclIbDevs[dev].maxQp;
  return ncclSuccess;
}

static ncclResult_t GetSocketAddr(union socketAddress* addr) {
  memcpy(addr, &ncclIbIfAddr, sizeof(*addr));
  return ncclSuccess;
}

#define MAX_REQUESTS 128

struct ncclIbQpInfo {
  uint32_t lid;
  uint8_t ib_port;
  uint32_t qpn;

  // For RoCE
  uint64_t spn;
  uint64_t iid;
  enum ibv_mtu mtu;

  // FIFO RDMA info
  uint32_t fifoRkey;
  uint64_t fifoAddr;
};

struct ncclIbHandle {
  union socketAddress connectAddr;
};

struct ncclIbVerbs {
  struct ibv_pd* pd;
  struct ibv_cq* cq;
};

struct ncclIbRequest {
  int used;
  int type;
  struct ncclIbVerbs* verbs;
  int done;
  int size;
  int free;
};

struct ncclIbListenComm {
  int dev;
  int fd;
};

struct ncclIbSendFifo {
  uint64_t addr;
  int      size;
  uint32_t seq;
  uint32_t rkey;
  uint32_t ready;
};

struct ncclIbSendComm {
  struct ncclIbVerbs verbs;
  struct ncclIbSendFifo fifo[MAX_REQUESTS];
  struct ncclIbRequest reqs[MAX_REQUESTS];
  uint32_t fifoHead;
  int fd;
  int ready;
  struct ibv_qp* qp;
  struct ibv_mr* fifoMr;
};

struct ncclIbGpuFlush {
  int enabled;
  int hostMem;
  struct ibv_mr* hostMr;
  struct ibv_sge sge;
  struct ibv_qp* qp;
};

struct ncclIbRemFifo {
  struct ncclIbSendFifo elems[MAX_REQUESTS];
  uint64_t addr;
  uint32_t rkey;
  uint32_t tail;
  uint32_t flags;
  struct ibv_mr* mr;
  struct ibv_sge sge;
};

struct ncclIbRecvComm {
  struct ncclIbVerbs verbs;
  struct ncclIbRemFifo remFifo;
  struct ncclIbRequest reqs[MAX_REQUESTS];
  int fd;
  int ready;
  struct ibv_qp* qp;
  struct ncclIbGpuFlush gpuFlush;
};

ncclResult_t ncclIbInitVerbs(struct ibv_context* ctx, struct ncclIbVerbs* verbs) {
  NCCLCHECK(wrap_ibv_alloc_pd(&verbs->pd, ctx));
  NCCLCHECK(wrap_ibv_create_cq(&verbs->cq, ctx, MAX_REQUESTS, NULL, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclIbDestroyVerbs(struct ncclIbVerbs* verbs) {
  NCCLCHECK(wrap_ibv_destroy_cq(verbs->cq));
  NCCLCHECK(wrap_ibv_dealloc_pd(verbs->pd));
  return ncclSuccess;
}

ncclResult_t ncclIbCreateQp(uint8_t ib_port, struct ncclIbVerbs* verbs, int access_flags, struct ibv_qp** qp) {
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.send_cq = verbs->cq;
  qpInitAttr.recv_cq = verbs->cq;
  qpInitAttr.qp_type = IBV_QPT_RC;
  qpInitAttr.cap.max_send_wr = MAX_REQUESTS;
  qpInitAttr.cap.max_recv_wr = MAX_REQUESTS;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = 0;
  NCCLCHECK(wrap_ibv_create_qp(qp, verbs->pd, &qpInitAttr));
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = 0;
  qpAttr.port_num = ib_port;
  qpAttr.qp_access_flags = access_flags;
  NCCLCHECK(wrap_ibv_modify_qp(*qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  return ncclSuccess;
}

ncclResult_t ncclIbRtrQp(struct ibv_qp* qp, struct ncclIbQpInfo* info) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = info->mtu;
  qpAttr.dest_qp_num = info->qpn;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;
  if (info->lid == 0) {
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.dgid.global.subnet_prefix = info->spn;
    qpAttr.ah_attr.grh.dgid.global.interface_id = info->iid;
    qpAttr.ah_attr.grh.flow_label = 0;
    qpAttr.ah_attr.grh.sgid_index = ncclParamIbGidIndex();
    qpAttr.ah_attr.grh.hop_limit = 255;
    qpAttr.ah_attr.grh.traffic_class = ncclParamIbTc();
  } else {
    qpAttr.ah_attr.is_global = 0;
    qpAttr.ah_attr.dlid = info->lid;
  }
  qpAttr.ah_attr.sl = ncclParamIbSl();
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = info->ib_port;
  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));
  return ncclSuccess;
}

ncclResult_t ncclIbRtsQp(struct ibv_qp* qp) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = ncclParamIbTimeout();
  qpAttr.retry_cnt = ncclParamIbRetryCnt();
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
  return ncclSuccess;
}


ncclResult_t ncclIbListen(int dev, void* opaqueHandle, void** listenComm) {
  struct ncclIbListenComm* comm;
  comm = malloc(sizeof(struct ncclIbListenComm));
  memset(comm, 0, sizeof(struct ncclIbListenComm));
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  NCCL_STATIC_ASSERT(sizeof(struct ncclIbHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclIbHandle size too large");
  comm->dev = dev;
  NCCLCHECK(GetSocketAddr(&(handle->connectAddr)));
  NCCLCHECK(createListenSocket(&comm->fd, &handle->connectAddr));
  *listenComm = comm;
  return ncclSuccess;
}

ncclResult_t ncclIbConnect(int dev, void* opaqueHandle, void** sendComm) {
  struct ncclIbSendComm* comm;
  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(struct ncclIbSendComm)));

  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  NCCLCHECK(connectAddress(&comm->fd, &handle->connectAddr));
  *sendComm = comm;

  // IB Setup
  struct ibv_context* ctx = ncclIbDevs[dev].context;
  NCCLCHECK(ncclIbInitVerbs(ctx, &comm->verbs));
  uint8_t ib_port = ncclIbDevs[dev].port;
  NCCLCHECK(ncclIbCreateQp(ib_port, &comm->verbs, IBV_ACCESS_REMOTE_WRITE, &comm->qp));

  // Send my QP Info to receiver through the socket. Hope this won't block.
  struct ibv_port_attr portAttr;
  NCCLCHECK(wrap_ibv_query_port(ctx, ib_port, &portAttr));
  struct ncclIbQpInfo qpInfo;
  qpInfo.ib_port = ib_port;
  qpInfo.qpn = comm->qp->qp_num;
  qpInfo.mtu = portAttr.active_mtu;

  // Prepare my fifo
  NCCLCHECK(wrap_ibv_reg_mr(&comm->fifoMr, comm->verbs.pd, comm->fifo, sizeof(struct ncclIbSendFifo)*MAX_REQUESTS, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ));
  qpInfo.fifoRkey = comm->fifoMr->rkey;
  qpInfo.fifoAddr = (uint64_t)comm->fifo;

  // RoCE support
  qpInfo.lid = portAttr.lid;
  if (qpInfo.lid) { // IB
    INFO(NCCL_NET,"NET/IB: Dev %d Port %d qpn %d mtu %d LID %d", dev, ib_port, qpInfo.qpn, qpInfo.mtu, qpInfo.lid);
  } else { // RoCE
    union ibv_gid gid;
    NCCLCHECK(wrap_ibv_query_gid(ctx, ib_port, ncclParamIbGidIndex(), &gid));
    qpInfo.spn = gid.global.subnet_prefix;
    qpInfo.iid = gid.global.interface_id;
    INFO(NCCL_NET,"NET/IB: Dev %d Port %d qpn %d mtu %d GID %ld (%lX/%lX)", dev, ib_port, qpInfo.qpn, qpInfo.mtu, ncclParamIbGidIndex(), qpInfo.spn, qpInfo.iid);
  }

  NCCLCHECK(socketSend(comm->fd, &qpInfo, sizeof(qpInfo)));
  return ncclSuccess;
}

NCCL_PARAM(IbGdrFlushDisable, "GDR_FLUSH_DISABLE", 0);

ncclResult_t ncclIbAccept(void* listenComm, void** recvComm) {
  struct ncclIbListenComm* lComm = (struct ncclIbListenComm*)listenComm;
  struct ncclIbRecvComm* rComm;
  NCCLCHECK(ncclIbMalloc((void**)&rComm, sizeof(struct ncclIbRecvComm)));

  struct sockaddr_in sockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  SYSCHECKVAL(accept(lComm->fd, (struct sockaddr*)&sockaddr, &socklen), "accept", rComm->fd);
  struct ncclIbQpInfo remQpInfo;
  NCCLCHECK(socketReceive(rComm->fd, &remQpInfo, sizeof(remQpInfo)));

  // IB setup
  struct ibv_context* ctx = ncclIbDevs[lComm->dev].context;
  uint8_t ib_port = ncclIbDevs[lComm->dev].port;
  struct ibv_port_attr portAttr;
  NCCLCHECK(wrap_ibv_query_port(ctx, ib_port, &portAttr));
  union ibv_gid gid;
  NCCLCHECK(wrap_ibv_query_gid(ctx, ib_port, ncclParamIbGidIndex(), &gid));

  // QP Creation
  NCCLCHECK(ncclIbInitVerbs(ctx, &rComm->verbs));
  NCCLCHECK(ncclIbCreateQp(ib_port, &rComm->verbs, IBV_ACCESS_REMOTE_WRITE, &rComm->qp));

  // Adjust the MTU
  remQpInfo.mtu = (enum ibv_mtu)MIN(remQpInfo.mtu, portAttr.active_mtu);

  // Setup QP
  struct ibv_qp* qp = rComm->qp;
  NCCLCHECK(ncclIbRtrQp(qp, &remQpInfo));
  NCCLCHECK(ncclIbRtsQp(qp));

  // Retain remote fifo info and prepare my RDMA ops
  rComm->remFifo.rkey = remQpInfo.fifoRkey;
  rComm->remFifo.addr = remQpInfo.fifoAddr;
  NCCLCHECK(wrap_ibv_reg_mr(&rComm->remFifo.mr, rComm->verbs.pd, &rComm->remFifo.elems, sizeof(struct ncclIbSendFifo)*MAX_REQUESTS, IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ));
  rComm->remFifo.sge.length = sizeof(struct ncclIbSendFifo);
  rComm->remFifo.sge.lkey = rComm->remFifo.mr->lkey;

#if USE_RDMA_SEND_INLINE
  // Determine whether the remFifo element data can be sent INLINE
  struct ibv_qp_attr attr;
  struct ibv_qp_init_attr init_attr;
  NCCLCHECK(wrap_ibv_query_qp(qp, &attr, IBV_QP_CAP, &init_attr));
  if (init_attr.cap.max_inline_data >= rComm->remFifo.sge.length) rComm->remFifo.flags = IBV_SEND_INLINE;
#endif

  // Allocate Flush dummy buffer for GPU Direct RDMA
  rComm->gpuFlush.enabled = (ncclIbGdrSupport(lComm->dev) == 0) && (ncclParamIbGdrFlushDisable() == 0) ? 1 : 0;
  if (rComm->gpuFlush.enabled) {
    NCCLCHECK(wrap_ibv_reg_mr(&rComm->gpuFlush.hostMr, rComm->verbs.pd, &rComm->gpuFlush.hostMem, sizeof(int), IBV_ACCESS_LOCAL_WRITE));
    rComm->gpuFlush.sge.addr = (uint64_t)&rComm->gpuFlush.hostMem;
    rComm->gpuFlush.sge.length = 1;
    rComm->gpuFlush.sge.lkey = rComm->gpuFlush.hostMr->lkey;
    NCCLCHECK(ncclIbCreateQp(ib_port, &rComm->verbs, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ, &rComm->gpuFlush.qp));
    struct ncclIbQpInfo localQpInfo = {
      .lid=portAttr.lid,
      .ib_port=ib_port,
      .qpn=rComm->gpuFlush.qp->qp_num,
      .spn=gid.global.subnet_prefix,
      .iid=gid.global.interface_id,
      .mtu=portAttr.active_mtu
    };
    NCCLCHECK(ncclIbRtrQp(rComm->gpuFlush.qp, &localQpInfo));
    NCCLCHECK(ncclIbRtsQp(rComm->gpuFlush.qp));
  }

  // Fill Handle
  struct ncclIbQpInfo qpInfo = {
    .lid=portAttr.lid,
    .ib_port=ib_port,
    .qpn=qp->qp_num,
    .spn=gid.global.subnet_prefix,
    .iid=gid.global.interface_id,
    .mtu=remQpInfo.mtu
  };

  NCCLCHECK(socketSend(rComm->fd, &qpInfo, sizeof(qpInfo)));
  *recvComm = rComm;
  return ncclSuccess;
}

ncclResult_t ncclIbGetRequest(struct ncclIbRequest* reqs, struct ncclIbRequest** req) {
  for (int i=0; i<MAX_REQUESTS; i++) {
    struct ncclIbRequest* r = reqs+i;
    if (r->used == 0) {
      r->used = 1;
      r->type = 0;
      r->verbs = NULL;
      r->done = 0;
      r->size = -1;
      r->free = 0;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/IB : unable to allocate requests");
  *req = NULL;
  return ncclInternalError;
}

ncclResult_t ncclSendCheck(struct ncclIbSendComm* comm) {
  struct ncclIbQpInfo remQpInfo;
  struct ibv_qp* qp = comm->qp;

  // Do not block on this receive, return if not ready.
  int bytes = 0;
  NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, comm->fd, &remQpInfo, sizeof(remQpInfo), &bytes));
  if (bytes == 0) return ncclSuccess; // Try again later
  NCCLCHECK(socketWait(NCCL_SOCKET_RECV, comm->fd, &remQpInfo, sizeof(remQpInfo), &bytes));

  NCCLCHECK(ncclIbRtrQp(qp, &remQpInfo));
  NCCLCHECK(ncclIbRtsQp(qp));
  comm->ready = 1;

  // Block until this is done. It *should* not block indefinitely.
  NCCLCHECK(socketSend(comm->fd, &comm->ready, sizeof(int)));

  return ncclSuccess;
}

ncclResult_t ncclRecvCheck(struct ncclIbRecvComm* comm) {
  // Do not block on this receive, return if not ready.
  int bytes = 0;
  NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, comm->fd, &comm->ready, sizeof(int), &bytes));
  if (bytes == 0) return ncclSuccess; // Try again later
  NCCLCHECK(socketWait(NCCL_SOCKET_RECV, comm->fd, &comm->ready, sizeof(int), &bytes));
  return ncclSuccess;
}

ncclResult_t ncclIbTest(void* request, int* done, int* size);

#define REG_ALIGN (4096)

ncclResult_t ncclIbRegMr(void* comm, void* data, int size, int type, void** mhandle) {
  struct ncclIbVerbs* verbs = (struct ncclIbVerbs*)comm;
  uint64_t addr = (uint64_t)data;
  unsigned flags;
  assert(size > 0);

  // Deregister / register
  uint64_t regAddr = addr & (~(REG_ALIGN-1));
  uint64_t regSize = addr+size - regAddr;
  regSize = ((regSize + REG_ALIGN-1) / REG_ALIGN ) * REG_ALIGN;
  struct ibv_mr* mr;
  flags = IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
  if (ncclParamIbPciRelaxedOrdering()) {
	flags |= IBV_ACCESS_RELAXED_ORDERING;
  }
  NCCLCHECK(wrap_ibv_reg_mr(&mr, verbs->pd, (void*)regAddr, regSize, flags));
  *mhandle = (void*)mr;
  TRACE(NCCL_INIT,"regAddr %lx size %ld rkey %x", regAddr, regSize, mr->rkey);
  return ncclSuccess;
}

ncclResult_t ncclIbDeregMr(void* comm, void* mhandle) {
  NCCLCHECK(wrap_ibv_dereg_mr((struct ibv_mr*)mhandle));
  return ncclSuccess;
}

ncclResult_t ncclIbIsend(void* sendComm, void* data, int size, void* mhandle, void** request) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm->ready == 0) NCCLCHECK(ncclSendCheck(comm));
  if (comm->ready == 0) { *request = NULL; return ncclSuccess; }

  struct ibv_mr* mr = (struct ibv_mr*)mhandle;

  // Wait for the receiver to have posted the corresponding receive
  volatile struct ncclIbSendFifo* slot = comm->fifo + (comm->fifoHead%MAX_REQUESTS);
  volatile uint32_t * readyPtr = &slot->ready;
  if (*readyPtr == 0) { *request = NULL; return ncclSuccess; }

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(comm->reqs, &req));
  req->verbs = &comm->verbs;
  req->size = size;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)req;

  struct ibv_sge sge;
  if (size == 0) {
    wr.sg_list = NULL;
    wr.num_sge = 0;
  } else {
    sge.addr=(uintptr_t)data; sge.length=(unsigned int)size; sge.lkey=mr->lkey;
    wr.sg_list = &sge;
    wr.num_sge = 1;
  }
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

#if USE_RDMA_WRITE
  __sync_synchronize(); // order the readyPtr load against rkey load below
  // Sanity checks to catch user collective call count/size mismatches
  // plus any potential programming errors
  if (size > slot->size || slot->size <= 0 || slot->addr == 0 || slot->rkey == 0 || slot->seq != comm->fifoHead) {
    WARN("NET/IB : collective mismatch error local size %d remote %d addr %lx rkey %x seq %x/%x",
        size, slot->size, slot->addr, slot->rkey, slot->seq, comm->fifoHead);
    return ncclInternalError;
  }
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.wr.rdma.remote_addr = slot->addr;
  wr.wr.rdma.rkey = slot->rkey;
  wr.imm_data = size; // Send the message size via imm_data
  __sync_synchronize();
#endif
  // We must clear slot->ready, but reset other fields to aid
  // debugging and sanity checks
  slot->ready = 0;
  slot->addr = 0ULL;
  slot->rkey = slot->size = slot->seq = 0;
  comm->fifoHead++;

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(comm->qp, &wr, &bad_wr));
  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, uint32_t rkey, uint64_t addr, int size) {
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(comm->reqs, &req));
  req->verbs = &comm->verbs;
  req->free = 1; // Not a user req ; free as soon as it is complete.
  wr.wr_id = (uint64_t)req;

  struct ncclIbSendFifo* localElem = comm->remFifo.elems + (comm->remFifo.tail % MAX_REQUESTS);
  localElem->addr = addr;
  localElem->rkey = rkey;
  localElem->ready = 1;
  localElem->size = size; // Sanity/Debugging
  localElem->seq = comm->remFifo.tail; // Sanity/Debugging
  wr.wr.rdma.remote_addr = comm->remFifo.addr + (comm->remFifo.tail % MAX_REQUESTS) * sizeof(struct ncclIbSendFifo);
  wr.wr.rdma.rkey = comm->remFifo.rkey;
  comm->remFifo.sge.addr = (uint64_t)localElem;
  wr.sg_list = &comm->remFifo.sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED | comm->remFifo.flags; // IBV_SEND_INLINE

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(comm->qp, &wr, &bad_wr));
  comm->remFifo.tail++;

  return ncclSuccess;
}

ncclResult_t ncclIbIrecv(void* recvComm, void* data, int size, void* mhandle, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm->ready == 0) NCCLCHECK(ncclRecvCheck(comm));
  if (comm->ready == 0) { *request = NULL; return ncclSuccess; }

  struct ibv_mr* mr = (struct ibv_mr*)mhandle;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(comm->reqs, &req));
  req->verbs = &comm->verbs;
  req->size = size;

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)req;

  struct ibv_sge sge;
  if (size == 0) {
    wr.sg_list = NULL;
    wr.num_sge = 0;
  } else {
    sge.addr=(uintptr_t)data; sge.length=(unsigned int)size; sge.lkey=mr->lkey;
    wr.sg_list = &sge;
    wr.num_sge = 1;
  }

  struct ibv_recv_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_recv(comm->qp, &wr, &bad_wr));
  *request = req;

  // Post to FIFO to notify sender
  NCCLCHECK(ncclIbPostFifo(comm, mr->rkey, (uint64_t)data, size));
  return ncclSuccess;
}

ncclResult_t ncclIbFlush(void* recvComm, void* data, int size, void* mhandle) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm->gpuFlush.enabled == 0 || size == 0) return ncclSuccess;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(comm->reqs, &req));
  req->verbs = &comm->verbs;
  struct ibv_mr* mr = (struct ibv_mr*)mhandle;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)req;

  wr.wr.rdma.remote_addr = (uint64_t)data;
  wr.wr.rdma.rkey = mr->rkey;
  wr.sg_list = &comm->gpuFlush.sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(comm->gpuFlush.qp, &wr, &bad_wr));

  int done = 0;
  while (done == 0) {
    NCCLCHECK((ncclResult_t)ncclIbTest(req, &done, NULL));
  }

  return ncclSuccess;
}

ncclResult_t ncclIbTest(void* request, int* done, int* size) {
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;
  *done = 0;

  while (1) {
    if (r->done == 1) {
      *done = 1;
      if (size) *size = r->size;
      r->used = 0;
      return ncclSuccess;
    }

    int wrDone = 0;
    struct ibv_wc wcs[4];
    NCCLCHECK(wrap_ibv_poll_cq(r->verbs->cq, 4, wcs, &wrDone));
    if (wrDone == 0) return ncclSuccess;

    for (int w=0; w<wrDone; w++) {
      struct ibv_wc *wc = wcs+w;
      if (wc->status != IBV_WC_SUCCESS) {
        WARN("NET/IB : Got completion with error %d, opcode %d, len %d, vendor err %d", wc->status, wc->opcode, wc->byte_len, wc->vendor_err);
        return ncclSystemError;
      }

      struct ncclIbRequest* doneReq = (struct ncclIbRequest*)wc->wr_id;
      if (doneReq) {
        if (wc->opcode == IBV_WC_RECV) {
          doneReq->size = wc->byte_len;
#if USE_RDMA_WRITE
        } else if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
          doneReq->size = wc->imm_data;
#endif
        }
        doneReq->done = 1;
        if (doneReq->free == 1) {
          // This is an internal (FIFO post) req. Free it immediately.
          doneReq->used = 0;
        }
      }
    }
  }
}

ncclResult_t ncclIbCloseSend(void* sendComm) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm) {
    close(comm->fd);
    if (comm->qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->qp));
    if (comm->fifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(comm->fifoMr));
    NCCLCHECK(ncclIbDestroyVerbs(&comm->verbs));
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclIbCloseRecv(void* recvComm) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm) {
    close(comm->fd);
    if (comm->qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->qp));
    if (comm->gpuFlush.enabled) {
      if (comm->gpuFlush.qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->gpuFlush.qp));
      if (comm->gpuFlush.hostMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(comm->gpuFlush.hostMr));
    }
    if (comm->remFifo.mr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(comm->remFifo.mr));
    NCCLCHECK(ncclIbDestroyVerbs(&comm->verbs));
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclIbCloseListen(void* listenComm) {
  struct ncclIbListenComm* comm = (struct ncclIbListenComm*)listenComm;
  if (comm) {
    close(comm->fd);
    free(comm);
  }
  return ncclSuccess;
}

ncclNet_t NCCL_PLUGIN_SYMBOL = {
  "IBext",
  ncclIbInit,
  ncclIbDevices,
  ncclIbGetProperties,
  ncclIbListen,
  ncclIbConnect,
  ncclIbAccept,
  ncclIbRegMr,
  ncclIbDeregMr,
  ncclIbIsend,
  ncclIbIrecv,
  ncclIbFlush,
  ncclIbTest,
  ncclIbCloseSend,
  ncclIbCloseRecv,
  ncclIbCloseListen
};

#include "sharp/api/version.h"
#include "sharp/api/sharp_coll.h"

struct ncclSharpRequest {
  void *sharpRequest;
  int size;
  int used;
};

struct ncclSharpCollComm {
  int rank;
  int nranks;
  void* recvComm;
  void* sendComm;

  struct ncclSharpRequest* reqs;

  struct sharp_coll_context* sharpCollContext;
  struct sharp_coll_comm* sharpCollComm;
};

struct ncclSharpMemHandle{
  void *mr;
  void *ncclIbMr;
  int type;
};

int ncclSharpAllGather(void *context, void *buf, int len) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)context;

  void* rMhandle, *sMhandle;
  NCCLCHECK(ncclIbRegMr(cComm->recvComm, buf, cComm->nranks*len, NCCL_PTR_HOST, &rMhandle));
  NCCLCHECK(ncclIbRegMr(cComm->sendComm, buf, cComm->nranks*len, NCCL_PTR_HOST, &sMhandle));
  int speer = cComm->rank;
  for (int i=0; i<cComm->nranks-1; i++) {
    void* srequest = NULL, *rrequest = NULL;
    int rpeer = (speer-1+cComm->nranks)%cComm->nranks;
    while (srequest == NULL || rrequest == NULL) {
       if (srequest == NULL) NCCLCHECK(ncclIbIsend(cComm->sendComm, ((char*)buf)+speer*len, len, sMhandle, &srequest));
       if (rrequest == NULL) NCCLCHECK(ncclIbIrecv(cComm->recvComm, ((char*)buf)+rpeer*len, len, rMhandle, &rrequest));
    }
    while (srequest || rrequest) {
      int done;
      if (rrequest) NCCLCHECK(ncclIbTest(rrequest, &done, NULL));
      if (done) rrequest = NULL;
      if (srequest) NCCLCHECK(ncclIbTest(srequest, &done, NULL));
      if (done) srequest = NULL;
    }
    speer = rpeer;
  }
  NCCLCHECK(ncclIbDeregMr(cComm->recvComm, rMhandle));
  NCCLCHECK(ncclIbDeregMr(cComm->sendComm, sMhandle));
  return 0;
}

struct ncclSharpInfo {
  uint64_t hostId;
  uint64_t jobId;
};

int ncclSharpOobBarrier(void *ctx) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)ctx;
  int* dummy;
  NCCLCHECK(ncclIbMalloc((void**)&dummy, cComm->nranks*sizeof(int)));
  NCCLCHECK(ncclSharpAllGather(ctx, dummy, sizeof(int)));
  free(dummy);
  return 0;
}

int ncclSharpOobGather(void *ctx, int root, void *sbuf, void *rbuf, int size) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)ctx;
  int nranks = cComm->nranks;
  void *tmp;
  NCCLCHECK(ncclIbMalloc(&tmp, nranks*size));
  memcpy((void*)((ptrdiff_t)tmp + size*cComm->rank), sbuf, size);
  NCCLCHECK(ncclSharpAllGather(cComm, tmp, size));
  if (cComm->rank == root) {
    memcpy(rbuf, tmp, nranks*size);
  }
  free(tmp);
  return 0;
}

int ncclSharpOobBcast(void *ctx, void *buf, int size, int root) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)ctx;
  void *tmp;
  NCCLCHECK(ncclIbMalloc(&tmp, size*cComm->nranks));
  if (cComm->rank == root) {
    memcpy((void*)((ptrdiff_t)tmp+size*cComm->rank), buf, size);
  }
  NCCLCHECK(ncclSharpAllGather(cComm, tmp, size));
  if (cComm->rank != root) {
    memcpy(buf, (void*)((ptrdiff_t)tmp+size*root), size);
  }
  free(tmp);
  return 0;
}

static __inline__ enum sharp_datatype typeConvert(ncclDataType_t type) {
  switch (type) {
    case ncclFloat16: return SHARP_DTYPE_FLOAT_SHORT;
    case ncclInt32: return SHARP_DTYPE_INT;
    case ncclUint32: return SHARP_DTYPE_UNSIGNED;
    case ncclFloat32: return SHARP_DTYPE_FLOAT;
    case ncclInt64: return SHARP_DTYPE_LONG;
    case ncclUint64: return SHARP_DTYPE_UNSIGNED_LONG;
    case ncclFloat64: return SHARP_DTYPE_DOUBLE;
    default: return SHARP_DTYPE_NULL;
  }
}

static __inline__ int typeSize(ncclDataType_t type) {
  switch (type) {
    case ncclFloat16: return 2;
    case ncclInt32: return 4;
    case ncclUint32: return 4;
    case ncclFloat32: return 4;
    case ncclInt64: return 8;
    case ncclUint64: return 8;
    case ncclFloat64: return 8;
    default:
      WARN("SHARP: unsupported data type\n");
      return -1;
  }
}

static __inline__ enum sharp_reduce_op opConvert(ncclRedOp_t op) {
  switch (op) {
    case ncclSum: return SHARP_OP_SUM;
    case ncclMax: return SHARP_OP_MAX;
    case ncclMin: return SHARP_OP_MIN;
    default: return SHARP_OP_NULL;
  }
}

ncclResult_t ncclSharpConnect(void* handles[], int nranks, int rank, void* listenComm, void** collComm) {
  struct ncclIbListenComm* lComm = (struct ncclIbListenComm*)listenComm;
  struct ncclSharpCollComm* cComm;
  NCCLCHECK(ncclIbMalloc((void**)&cComm, sizeof(struct ncclSharpCollComm)));
  NCCLCHECK(ncclIbMalloc((void**)&cComm->reqs, sizeof(struct ncclSharpRequest)*MAX_REQUESTS));

  cComm->nranks = nranks;
  cComm->rank = rank;
  if (cComm->rank == -1) {
    WARN("Could not determine my rank\n");
    return ncclInternalError;
  }
  int next = (cComm->rank + 1) % nranks;
  NCCLCHECK(ncclIbConnect(lComm->dev, handles[next], &cComm->sendComm));
  NCCLCHECK(ncclIbAccept(listenComm, &cComm->recvComm)); // From prev

  struct ncclSharpInfo* allInfo;
  pid_t pid = getpid();
  pthread_t tid = pthread_self();
  NCCLCHECK(ncclIbMalloc((void**)&allInfo, sizeof(struct ncclSharpInfo)*nranks));
  allInfo[cComm->rank].hostId = gethostid();
  allInfo[cComm->rank].jobId = (((uint64_t)allInfo[cComm->rank].hostId << 32) | ((pid ^ tid) ^ rand()));
  NCCLCHECK(ncclSharpAllGather(cComm, allInfo, sizeof(struct ncclSharpInfo)));

  // Find my local rank;
  int localRank = 0;
  for (int i=0; i<cComm->rank; i++) {
    if (allInfo[cComm->rank].hostId == allInfo[i].hostId) {
      localRank++;
    }
  }
  uint64_t jobId = allInfo[0].jobId;
  free(allInfo);

  struct sharp_coll_init_spec init_spec = {0};
  init_spec.progress_func  = NULL;
  init_spec.job_id = jobId;
  init_spec.world_rank = cComm->rank;
  init_spec.world_size = nranks;
  init_spec.world_local_rank = 0;
  init_spec.enable_thread_support = 1;
  init_spec.group_channel_idx = 0;

  init_spec.oob_colls.barrier = ncclSharpOobBarrier;
  init_spec.oob_colls.bcast = ncclSharpOobBcast;
  init_spec.oob_colls.gather = ncclSharpOobGather;
  init_spec.oob_ctx = cComm;

  init_spec.config = sharp_coll_default_config;
  init_spec.config.user_progress_num_polls = 10000000;

  char devName[MAXNAMESIZE];
  snprintf(devName, MAXNAMESIZE, "%s:%d", ncclIbDevs[lComm->dev].devName, ncclIbDevs[lComm->dev].port);
  init_spec.config.ib_dev_list = devName;

  int ret = sharp_coll_init(&init_spec, &cComm->sharpCollContext);

  INFO(NCCL_INIT, "Sharp rank %d/%d initialized on %s", cComm->rank, nranks, devName);

  if (ret < 0) {
    WARN("NET/IB :SHARP coll init error: %s(%d)\n", sharp_coll_strerror(ret), ret);
    return ncclInternalError;
  }

  struct sharp_coll_comm_init_spec comm_spec;
  comm_spec.rank = cComm->rank;
  comm_spec.size = nranks;
  comm_spec.oob_ctx = cComm;
  comm_spec.group_world_ranks = NULL;

  ret = sharp_coll_comm_init(cComm->sharpCollContext, &comm_spec, &cComm->sharpCollComm);
  if (ret < 0) {
    WARN("SHARP group create failed: %s(%d)\n", sharp_coll_strerror(ret), ret);
    return ncclInternalError;
  }

  *collComm = cComm;
  return ncclSuccess;
}

ncclResult_t ncclSharpReduceSupport(ncclDataType_t dataType, ncclRedOp_t redOp, int* supported) {
  *supported = ((typeConvert(dataType) != SHARP_DTYPE_NULL) && (opConvert(redOp) != SHARP_OP_NULL));
  return ncclSuccess;
}

ncclResult_t ncclSharpRegMr(void* collComm, void* data, int size, int type, void** mhandle) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)collComm;

  struct ncclSharpMemHandle *mh;
  NCCLCHECK(ncclIbMalloc((void**)&mh, sizeof(struct ncclSharpMemHandle)));

  mh->type = type;
  if (SHARP_COLL_SUCCESS != sharp_coll_reg_mr(cComm->sharpCollContext, data, size, &(mh->mr)))  {
    WARN("SHARP regmr failed\n");
    return ncclSystemError;
  }
  TRACE(NCCL_INIT,"sharpRegAddr %lx size %ld handle %x", data, size, mh->mr);

  NCCLCHECK(ncclIbRegMr(cComm->recvComm, data, size, type, &mh->ncclIbMr));

  *mhandle = mh;
  return ncclSuccess;
}

ncclResult_t ncclSharpDeregMr(void* collComm, void* mhandle) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)collComm;
  struct ncclSharpMemHandle *mh = (struct ncclSharpMemHandle *)mhandle;

  if (SHARP_COLL_SUCCESS != sharp_coll_dereg_mr(cComm->sharpCollContext, mh->mr)) {
    WARN("SHARP deregmr failed\n");
  }

  NCCLCHECK(ncclIbDeregMr(cComm->recvComm, mh->ncclIbMr));

  free(mh);
  return ncclSuccess;
}

ncclResult_t ncclSharpGetRequest(struct ncclSharpRequest* reqs, struct ncclSharpRequest** req) {
  for (int i=0; i<MAX_REQUESTS; i++) {
    struct ncclSharpRequest* r = reqs+i;
    if (r->used == 0) {
      r->used = 1;
      r->sharpRequest = NULL;
      r->size = 0;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("SHARP : unable to allocate request");
  *req = NULL;
  return ncclInternalError;
}

ncclResult_t ncclSharpIallreduce(void* collComm, void* sendData, void* recvData, int count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)collComm;

  enum sharp_datatype sharp_type = typeConvert(dataType);
  if (sharp_type == SHARP_DTYPE_NULL) {
    WARN("SHARP: unsupported data type\n");
    return ncclInternalError;
  }

  enum sharp_reduce_op op_type = opConvert(redOp);
  if (op_type == SHARP_OP_NULL) {
    WARN("SHARP: unsupported reduce operation\n");
    return ncclInternalError;
  }

  int dt_size = typeSize(dataType);
  struct ncclSharpMemHandle *mr_sbuf = (struct ncclSharpMemHandle*)sendMhandle;
  struct ncclSharpMemHandle *mr_rbuf = (struct ncclSharpMemHandle*)recvMhandle;

  struct ncclSharpRequest* req;
  NCCLCHECK(ncclSharpGetRequest(cComm->reqs, &req));

  struct sharp_coll_reduce_spec reduce_spec;

  reduce_spec.sbuf_desc.buffer.ptr = sendData;
  reduce_spec.sbuf_desc.buffer.length = count * dt_size;
  reduce_spec.sbuf_desc.buffer.mem_handle = mr_sbuf->mr;
  reduce_spec.sbuf_desc.type = SHARP_DATA_BUFFER;
  reduce_spec.sbuf_desc.mem_type = (mr_sbuf->type == NCCL_PTR_CUDA ? SHARP_MEM_TYPE_CUDA:SHARP_MEM_TYPE_HOST);

  reduce_spec.rbuf_desc.buffer.ptr = recvData;
  reduce_spec.rbuf_desc.buffer.length = count * dt_size;
  reduce_spec.rbuf_desc.buffer.mem_handle = mr_rbuf->mr;
  reduce_spec.rbuf_desc.type = SHARP_DATA_BUFFER;
  reduce_spec.rbuf_desc.mem_type = (mr_rbuf->type == NCCL_PTR_CUDA ? SHARP_MEM_TYPE_CUDA:SHARP_MEM_TYPE_HOST);

  reduce_spec.length = count;
  reduce_spec.dtype = sharp_type;
  reduce_spec.op = op_type;
  reduce_spec.aggr_mode = SHARP_AGGREGATION_NONE;

#if BLOCKING==0
  if (SHARP_COLL_SUCCESS != sharp_coll_do_allreduce_nb(cComm->sharpCollComm, &reduce_spec, &req->sharpRequest)) {
    WARN("SHARP allreduce failed\n");
  }
  req->size =  count * dt_size;
#else
  if (SHARP_COLL_SUCCESS != sharp_coll_do_allreduce(cComm->sharpCollComm, &reduce_spec)) {
    WARN("SHARP allreduce failed\n");
  }
  req->sharpRequest = (void *) 0xabababab;
  req->size =  count * dt_size;
#endif

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclSharpFlush(void* collComm, void* data, int size, void* mhandle) {
  struct ncclSharpCollComm *cComm = (struct ncclSharpCollComm*)collComm;
  struct ncclSharpMemHandle *mh = (struct ncclSharpMemHandle *)mhandle;

  return ncclIbFlush(cComm->recvComm, data, size, mh->ncclIbMr);
}

ncclResult_t ncclSharpTest(void* request, int* done, int* size) {
  struct ncclSharpRequest* req = (struct ncclSharpRequest*)request;

#if BLOCKING==0
  *done = sharp_coll_req_test(req->sharpRequest);
  if (*done){
    sharp_coll_req_free(req->sharpRequest);
    *size = req->size;
    req->used = 0;
  } else {
    *done = 0;
  }
#else
  if (req->size != -1) {
    *done = 1;
    *size = req->size;
    req->used = 0;
  } else {
     *done = 0;
  }
#endif
  return ncclSuccess;
}

ncclResult_t ncclSharpCloseColl(void* collComm) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)collComm;

  sharp_coll_comm_destroy(cComm->sharpCollComm);
  sharp_coll_finalize(cComm->sharpCollContext);

  NCCLCHECK(ncclIbCloseRecv(cComm->recvComm));
  NCCLCHECK(ncclIbCloseSend(cComm->sendComm));
  free(cComm);
  return ncclSuccess;
}

ncclCollNet_t NCCL_COLLNET_PLUGIN_SYMBOL = {
  "SHARP",
  ncclIbInit,
  ncclIbSharpDevices,
  ncclIbGetProperties,
  ncclIbListen,
  ncclSharpConnect,
  ncclSharpReduceSupport,
  ncclSharpRegMr,
  ncclSharpDeregMr,
  ncclSharpIallreduce,
  ncclSharpFlush,
  ncclSharpTest,
  ncclSharpCloseColl,
  ncclIbCloseListen
};
