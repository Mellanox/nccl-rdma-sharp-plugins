/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
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
#include "p2p_plugin.h"
#include "core.h"
#include "socket.h"
#include "utils.h"
#include "param.h"

#include <assert.h>
#include "ibvwrap.h"

#define USE_RDMA_WRITE 1
#define USE_RDMA_SEND_INLINE 0
#define MAXNAMESIZE 64
static char ncclIbIfName[MAX_IF_NAME_SIZE];
static union socketAddress ncclIbIfAddr;

static int ncclNIbDevs = -1;

pthread_mutex_t ncclIbLock = PTHREAD_MUTEX_INITIALIZER;

NCCL_PARAM(IbGidIndex, "IB_GID_INDEX", 0);
NCCL_PARAM(IbTimeout, "IB_TIMEOUT", 14);
NCCL_PARAM(IbRetryCnt, "IB_RETRY_CNT", 7);
NCCL_PARAM(IbSl, "IB_SL", 0);
NCCL_PARAM(IbTc, "IB_TC", 0);
NCCL_PARAM(IbPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 0);
NCCL_PARAM(IbArThreshold, "IB_AR_THRESHOLD", 8192);

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

extern ncclDebugLogger_t pluginLogFunction;

ncclResult_t ncclIbInit(ncclDebugLogger_t logFunction) {
  if (ncclParamIbDisable()) return ncclInternalError;
  if (ncclNIbDevs == -1) {
    if (ncclParamIbPciRelaxedOrdering() && !IBV_ACCESS_RELAXED_ORDERING) {
      WARN("NET/IB: PCI relaxed order memory access requested but not supported");
    }
  }
  return nccl_p2p_ib_init(&ncclNIbDevs, ncclIbDevs, ncclIbIfName, &ncclIbIfAddr, &ncclIbAsyncThread, logFunction);
}

ncclResult_t ncclIbDevices(int* ndev) {
  *ndev = ncclNIbDevs;
  return ncclSuccess;
}

ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props)
{
  return nccl_p2p_ib_get_properties(ncclIbDevs, dev, props);
}

static ncclResult_t GetSocketAddr(union socketAddress* addr) {
  memcpy(addr, &ncclIbIfAddr, sizeof(*addr));
  return ncclSuccess;
}

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
  uint64_t pad[1]; // Pad FIFO element size to be 32-bytes
};

struct ncclIbSendComm {
  struct ncclIbDev* dev;
  uint64_t pad[3]; // Pad SendFifo base to be 32-byte aligned
  struct ncclIbSendFifo fifo[MAX_REQUESTS];
  struct ncclIbRequest reqs[MAX_REQUESTS];
  uint32_t fifoHead;
  int fd;
  int ready;
  struct ibv_qp* qp;
  struct ibv_mr* fifoMr;
};
// The SendFifo needs to be 32-byte aligned and each element needs
// to be a 32-byte multiple, so that an entry does not get split and
// written out of order when IB Relaxed Ordering is enabled
static_assert((offsetof(struct ncclIbSendComm, fifo) % 32) == 0, "ncclIbSendComm fifo must be 32-byte aligned");
static_assert((sizeof(struct ncclIbSendFifo) % 32) == 0, "ncclIbSendFifo element size must be 32-byte multiples");

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
  struct ncclIbDev* dev;
  struct ncclIbRemFifo remFifo;
  struct ncclIbRequest reqs[MAX_REQUESTS];
  int fd;
  int ready;
  struct ibv_qp* qp;
  struct ncclIbGpuFlush gpuFlush;
};

pthread_mutex_t ibDevLock = PTHREAD_MUTEX_INITIALIZER;

ncclResult_t ncclIbUseDev(int dev, struct ncclIbDev** ibDevPtr) {
  pthread_mutex_lock(&ibDevLock);
  struct ncclIbDev* ibDev = ncclIbDevs+dev;
  if (ibDev->verbs.refCount == 0) {
    NCCLCHECK(wrap_ibv_alloc_pd(&ibDev->verbs.pd, ibDev->context));
    NCCLCHECK(wrap_ibv_create_cq(&ibDev->verbs.cq, ibDev->context, MAX_REQUESTS, NULL, NULL, 0));
  }
  ibDev->verbs.refCount++;
  *ibDevPtr = ibDev;
  pthread_mutex_unlock(&ibDevLock);
  return ncclSuccess;
}

ncclResult_t ncclIbFreeDev(struct ncclIbDev* ibDev) {
  pthread_mutex_lock(&ibDevLock);
  ibDev->verbs.refCount--;
  if (ibDev->verbs.refCount == 0) {
    NCCLCHECK(wrap_ibv_destroy_cq(ibDev->verbs.cq));
    NCCLCHECK(wrap_ibv_dealloc_pd(ibDev->verbs.pd));
  }
  pthread_mutex_unlock(&ibDevLock);
  return ncclSuccess;
}

ncclResult_t ncclIbCreateQp(uint8_t ib_port, struct ncclIbVerbs* verbs, int access_flags, struct ibv_qp** qp) {
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.send_cq = verbs->cq;
  qpInitAttr.recv_cq = verbs->cq;
  qpInitAttr.qp_type = IBV_QPT_RC;
  // We might send 2 requests per send (RDMA_WRITE+RDMA_WRITE_WITH_IMM)
  qpInitAttr.cap.max_send_wr = 2*MAX_REQUESTS;
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
  static_assert(sizeof(struct ncclIbHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclIbHandle size too large");
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
  NCCLCHECK(ncclIbUseDev(dev, &comm->dev));
  uint8_t ib_port = ncclIbDevs[dev].port;
  NCCLCHECK(ncclIbCreateQp(ib_port, &comm->dev->verbs, IBV_ACCESS_REMOTE_WRITE, &comm->qp));

  // Send my QP Info to receiver through the socket. Hope this won't block.
  struct ibv_port_attr portAttr;
  NCCLCHECK(wrap_ibv_query_port(ctx, ib_port, &portAttr));
  struct ncclIbQpInfo qpInfo;
  qpInfo.ib_port = ib_port;
  qpInfo.qpn = comm->qp->qp_num;
  qpInfo.mtu = portAttr.active_mtu;

  // Prepare my fifo
  NCCLCHECK(wrap_ibv_reg_mr(&comm->fifoMr, comm->dev->verbs.pd, comm->fifo, sizeof(struct ncclIbSendFifo)*MAX_REQUESTS, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ));
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
  NCCLCHECK(socketRecv(rComm->fd, &remQpInfo, sizeof(remQpInfo)));

  // IB setup
  struct ibv_context* ctx = ncclIbDevs[lComm->dev].context;
  uint8_t ib_port = ncclIbDevs[lComm->dev].port;
  struct ibv_port_attr portAttr;
  NCCLCHECK(wrap_ibv_query_port(ctx, ib_port, &portAttr));
  union ibv_gid gid;
  NCCLCHECK(wrap_ibv_query_gid(ctx, ib_port, ncclParamIbGidIndex(), &gid));

  // QP Creation
  NCCLCHECK(ncclIbUseDev(lComm->dev, &rComm->dev));
  NCCLCHECK(ncclIbCreateQp(ib_port, &rComm->dev->verbs, IBV_ACCESS_REMOTE_WRITE, &rComm->qp));

  // Adjust the MTU
  remQpInfo.mtu = (enum ibv_mtu)MIN(remQpInfo.mtu, portAttr.active_mtu);

  // Setup QP
  struct ibv_qp* qp = rComm->qp;
  NCCLCHECK(ncclIbRtrQp(qp, &remQpInfo));
  NCCLCHECK(ncclIbRtsQp(qp));

  // Retain remote fifo info and prepare my RDMA ops
  rComm->remFifo.rkey = remQpInfo.fifoRkey;
  rComm->remFifo.addr = remQpInfo.fifoAddr;
  NCCLCHECK(wrap_ibv_reg_mr(&rComm->remFifo.mr, rComm->dev->verbs.pd, &rComm->remFifo.elems, sizeof(struct ncclIbSendFifo)*MAX_REQUESTS, IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ));
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
  rComm->gpuFlush.enabled = (nccl_p2p_gdr_support(lComm->dev) == 0) && (ncclParamIbGdrFlushDisable() == 0) ? 1 : 0;
  if (rComm->gpuFlush.enabled) {
    NCCLCHECK(wrap_ibv_reg_mr(&rComm->gpuFlush.hostMr, rComm->dev->verbs.pd, &rComm->gpuFlush.hostMem, sizeof(int), IBV_ACCESS_LOCAL_WRITE));
    rComm->gpuFlush.sge.addr = (uint64_t)&rComm->gpuFlush.hostMem;
    rComm->gpuFlush.sge.length = 1;
    rComm->gpuFlush.sge.lkey = rComm->gpuFlush.hostMr->lkey;
    NCCLCHECK(ncclIbCreateQp(ib_port, &rComm->dev->verbs, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ, &rComm->gpuFlush.qp));
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
  static_assert(offsetof(struct ncclIbSendComm, dev) == offsetof(struct ncclIbRecvComm, dev), "Send and recv comms must have dev at the same offset");
  struct ncclIbSendComm* c = (struct ncclIbSendComm*)comm;
  struct ncclIbVerbs* verbs = &c->dev->verbs;
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
  req->verbs = &comm->dev->verbs;
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

  int useAr = 0;
  if (size > ncclParamIbArThreshold()) {
    useAr = 1;
  }
#if USE_RDMA_WRITE
  __sync_synchronize(); // order the readyPtr load against rkey load below
  // Sanity checks to catch user collective call count/size mismatches
  // plus any potential programming errors
  if (size > slot->size || slot->size < 0 || slot->addr == 0 || slot->rkey == 0 || slot->seq != comm->fifoHead) {
    WARN("NET/IB : collective mismatch error local size %d remote %d addr %lx rkey %x seq %x/%x",
        size, slot->size, slot->addr, slot->rkey, slot->seq, comm->fifoHead);
    return ncclInternalError;
  }
  wr.opcode = useAr ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_WRITE_WITH_IMM;
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

#if USE_RDMA_WRITE
  // When using adaptive routing, send the bulk of the data first as an
  // RDMA_WRITE, then a 0-byte RDMA_WRITE_WITH_IMM to trigger a remote
  // completion.
  if (useAr) {
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.sg_list = NULL;
    wr.num_sge = 0;
    wr.send_flags &= ~IBV_SEND_SIGNALED;
    NCCLCHECK(wrap_ibv_post_send(comm->qp, &wr, &bad_wr));
  }
#endif
  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, uint32_t rkey, uint64_t addr, int size) {
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(comm->reqs, &req));
  req->verbs = &comm->dev->verbs;
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
  req->verbs = &comm->dev->verbs;
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

ncclResult_t ncclIbIflush(void* recvComm, void* data, int size, void* mhandle, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm->gpuFlush.enabled == 0 || size == 0) return ncclSuccess;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(comm->reqs, &req));
  req->verbs = &comm->dev->verbs;
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

  *request = req;
  return ncclSuccess;
}

#define NUM_WC 8
ncclResult_t ncclIbTest(void* request, int* done, int* size) {
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;
  *done = 0;

    if (r->done == 1) {
      *done = 1;
      if (size) *size = r->size;
      r->used = 0;
      return ncclSuccess;
    }
  int wcDone = 0;
  do {
    struct ibv_wc wcs[NUM_WC];
    NCCLCHECK(wrap_ibv_poll_cq(r->verbs->cq, NUM_WC, wcs, &wcDone));

    for (int w=0; w<wcDone; w++) {
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
  } while (wcDone == NUM_WC);

  if (r->done == 1) {
    *done = 1;
    if (size) *size = r->size;
    r->used = 0;
  }
  return ncclSuccess;
}

ncclResult_t ncclIbCloseSend(void* sendComm) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm) {
    close(comm->fd);
    if (comm->qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->qp));
    if (comm->fifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(comm->fifoMr));
    NCCLCHECK(ncclIbFreeDev(comm->dev));
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
    NCCLCHECK(ncclIbFreeDev(comm->dev));
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

ncclNet_t ibPlugin = {
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
  ncclIbIflush,
  ncclIbTest,
  ncclIbCloseSend,
  ncclIbCloseRecv,
  ncclIbCloseListen
};
