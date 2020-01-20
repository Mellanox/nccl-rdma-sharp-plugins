
/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See LICENSE.txt for license information
 ************************************************************************/


#include "nccl.h"
#include "nccl_net.h"
#include "core.h"
#include "socket.h"
#include "utils.h"
#include "param.h"
#include "ibvwrap.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>

#include <ucp/api/ucp.h>

#define UCXCHECK(cmd) do {                               \
  int e = cmd;                                           \
  if( UCS_OK != e ) {                                    \
    WARN("Failed: UCX error %s:%d '%d' %s\n",   \
        __FILE__,__LINE__, e, ucs_status_string(e));     \
    return ncclInternalError;                            \
  }                                                      \
} while(0)

#define UCXCHECK_VOID(cmd) do {                               \
  int e = cmd;                                           \
  if( UCS_OK != e ) {                                    \
    WARN("Failed: UCX error %s:%d '%d' %s\n",   \
        __FILE__,__LINE__, e, ucs_status_string(e));     \
  }                                                      \
} while(0)

static const ucp_tag_t tag  = 0xABADBABE;
static const ucp_tag_t tag_mask = 0xFFFFFFFFFFFFFFFF;

static int ncclNIbDevs = -1;
extern int ncclNSharpDevs;

#define MAXNAMESIZE 64
#define IB_DEVICE_SYSFS_FMT "/sys/class/infiniband/%s/device/%s"

struct userIbDev {
  char devName[MAXNAMESIZE];
  uint16_t port_en;
};

struct ucx_mhandle {
  ucp_mem_h ucp_memh;
  ucp_rkey_h rkey;
};
typedef struct ucx_mhandle ucx_mhandle;

#define MAX_IB_DEVS 16
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
struct userIbDev userIbDevs[MAX_IB_DEVS];

ncclResult_t nccl_ucx_devices(int* ndev) {
  *ndev = ncclNIbDevs;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_pci_path(int dev, char** path) {
  char devicepath[PATH_MAX];
  snprintf(devicepath, PATH_MAX, "/sys/class/infiniband/%s/device", ncclIbDevs[dev].devName);
  *path = realpath(devicepath, NULL);
  if (*path == NULL) {
    WARN("Could not find real path of %s", devicepath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t nccl_ucx_gdr_support(int ibDev) {
  static int moduleLoaded = -1;
  if (moduleLoaded == -1) {
    moduleLoaded = (access("/sys/kernel/mm/memory_peers/nv_mem/version", F_OK) == -1) ? 0 : 1;
  }
  if (moduleLoaded == 0) return ncclSystemError;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_ptr_support(int dev, int* supported_types) {
  *supported_types = NCCL_PTR_HOST;

  if (nccl_ucx_gdr_support(dev) != ncclSuccess) {
    INFO(NCCL_NET,"NET/UCX : GPU Direct RDMA Disabled for HCA %d '%s' (no module)", dev, ncclIbDevs[dev].devName);
    return ncclSuccess;
  }
  *supported_types |= NCCL_PTR_CUDA;
  return ncclSuccess;
}

pthread_mutex_t nccl_ucx_lock = PTHREAD_MUTEX_INITIALIZER;

struct ucx_listen_handle {
  union socketAddress connectAddr;
  ucp_tag_t tag;
};
typedef struct ucx_listen_handle ucx_listen_handle;

struct ucx_listen_comm {
  int dev;
  int fd;
  ucp_context_h ctx;
  ucp_worker_h worker;
  ucp_tag_t tag;
};
typedef struct ucx_listen_comm ucx_listen_comm;

struct connect_msg {
  size_t addr_len;
};
typedef struct connect_msg connect_msg;

struct ucx_request {
  int completed;
  int size;
  ucp_worker_h worker;
};
typedef struct ucx_request ucx_request;

struct nccl_ucx_worker {
  ucp_context_h ctx;
  ucp_worker_h worker;
  int count;
};
static struct nccl_ucx_worker workers[MAX_IB_DEVS];

struct ucx_gpu_flush {
  int enabled;
  int hostMem;
  ucp_ep_h flush_ep;
};

struct ucx_ctx {
  ucp_context_h ucp_ctx;
  struct ucx_gpu_flush gpuFlush;
};
typedef struct ucx_ctx ucx_ctx;

struct ucx_send_comm {
  ucp_context_h ctx;
  struct ucx_gpu_flush gpuFlush;
  ucp_worker_h worker;
  ucp_ep_h ep;
  ucp_tag_t tag;
  ucp_tag_t ctag;
  int fd;
  int ready;
  uint32_t fifo_head;
  uint32_t fifo_tail;
  ucp_mem_h fifo_memh;
};
typedef struct ucx_send_comm ucx_send_comm;

struct ucx_recv_comm {
  ucp_context_h ctx;
  struct ucx_gpu_flush gpuFlush;
  ucp_worker_h worker;
  ucp_ep_h ep;
  ucp_tag_t tag;
  ucp_tag_t ctag;
  int fd;
  int ready;
  uint64_t rem_tail_addr;
  uint32_t tail;
  ucp_rkey_h rkey;
  connect_msg *msg;
  ucx_request *connect_req;
};
typedef struct ucx_recv_comm ucx_recv_comm;

static void request_init(void *request) {
  struct ucx_request *req = (struct ucx_request *)request;
  req->completed = 0;
}

static void send_handler(void *request, ucs_status_t status) {
  struct ucx_request *req = (struct ucx_request *)request;
  req->completed = 1;
}

static void recv_handler(void *request, ucs_status_t status, ucp_tag_recv_info_t *info) {
  struct ucx_request *req = (struct ucx_request *)request;
  req->completed = 1;
}

static union socketAddress nccl_ucx_if_addr;
static char if_name[MAX_IF_NAME_SIZE];

static ncclResult_t get_socket_addr(union socketAddress *addr) {
  memcpy(addr, &nccl_ucx_if_addr, sizeof(*addr));
  return ncclSuccess;
}

static ncclResult_t ucx_init_context(ucp_context_h *ctx, int dev) {
  ucp_params_t ucp_params;
  ucp_config_t *config;
  char ucx_dev_name[PATH_MAX];

  snprintf(ucx_dev_name, PATH_MAX, "%s:%d", ncclIbDevs[dev].devName, ncclIbDevs[dev].port);
  UCXCHECK(ucp_config_read("NCCL", NULL, &config));
  UCXCHECK(ucp_config_modify(config, "NET_DEVICES", ucx_dev_name));
  memset(&ucp_params, 0, sizeof(ucp_params));
  ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES |
                          UCP_PARAM_FIELD_REQUEST_SIZE |
                          UCP_PARAM_FIELD_REQUEST_INIT;
  ucp_params.features = UCP_FEATURE_TAG | UCP_FEATURE_RMA;
  ucp_params.request_size = sizeof(struct ucx_request);
  ucp_params.request_init = request_init;
  UCXCHECK(ucp_init(&ucp_params, config, ctx));
  ucp_config_release(config);
}

static ncclResult_t ucx_init_worker(ucp_context_h ctx, ucp_worker_h *worker) {
  ucp_worker_params_t worker_params;
  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
  UCXCHECK(ucp_worker_create(ctx, &worker_params, worker));
}

static ncclResult_t ucx_worker_get_netaddress(ucp_worker_h worker, ucp_address_t **address, size_t *address_length) {
  ucp_worker_attr_t attr;
  attr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS |
                    UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS;
  attr.address_flags = UCP_WORKER_ADDRESS_FLAG_NET_ONLY;
  UCXCHECK(ucp_worker_query(worker, &attr));
  *address = malloc(attr.address_length);
  memcpy(*address, attr.address, attr.address_length);
  *address_length = attr.address_length;
  free(attr.address);
  return ncclSuccess;
}

#define UCX_SHARED_WORKER
static ncclResult_t ucx_get_ctx_and_worker(int dev, ucp_context_h *ctx, ucp_worker_h *worker, ucp_tag_t *newtag) {
  pthread_mutex_lock(&nccl_ucx_lock);
#ifdef UCX_SHARED_WORKER
  if (ncclNIbDevs < dev) {
    WARN("Device index is too large");
    return ncclSystemError;
  }
  if (workers[dev].count == 0) {
    ucx_init_context(&workers[dev].ctx, dev);
    ucx_init_worker(workers[dev].ctx, &workers[dev].worker);
  }
  *ctx = workers[dev].ctx;
  *worker = workers[dev].worker;
  if (newtag != NULL) {
    *newtag = tag + workers[dev].count;
  }
  ucp_worker_progress(*worker);
  workers[dev].count++;
#else
  ucx_init_context(ctx, dev);
  ucx_init_worker(*ctx, worker);
  if (newtag != NULL) {
    *newtag = tag;
  }
#endif
  pthread_mutex_unlock(&nccl_ucx_lock);
  return ncclSuccess;
}

NCCL_PARAM(UCXDisable, "UCX_DISABLE", 0);
extern ncclDebugLogger_t pluginLogFunction;

ncclResult_t nccl_ucx_init(ncclDebugLogger_t logFunction) {
  struct timeval tval;
  gettimeofday(&tval, NULL);
  srand((int) tval.tv_usec);

  if (ncclParamUCXDisable()) return ncclInternalError;

  if (ncclNIbDevs == -1) {
    pthread_mutex_lock(&nccl_ucx_lock);
    wrap_ibv_fork_init();
    if (ncclNIbDevs == -1) {
      ncclNIbDevs = 0;
      ncclNSharpDevs = 0;
      if (findInterfaces(if_name, &nccl_ucx_if_addr, MAX_IF_NAME_SIZE, 1) != 1){
        WARN("NET/UCX : No IP interface found.");
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
          WARN("NET/UCX : Unable to open device %s", devices[d]->name);
          continue;
        }
        int found = 0;
        struct ibv_device_attr devAttr;
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
          WARN("NET/UCX : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context)) { return ncclInternalError; }
          continue;
        }
        for (int port = 1; port <= devAttr.phys_port_cnt; port++) {
          struct ibv_port_attr portAttr;
          long vendorId, devId;
          if (ncclSuccess != wrap_ibv_query_port(context, port, &portAttr)) {
            WARN("NET/UCX : Unable to query port %d", port);
            continue;
          }
          if (portAttr.state != IBV_PORT_ACTIVE) continue;
          if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND
              && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) continue;

          // check against user specified HCAs/ports
          if (! (matchIfList(devices[d]->name, port, userIfs, nUserIfs) ^ searchNot)) {
            continue;
          }
          TRACE(NCCL_INIT|NCCL_NET,"NET/UCX: [%d] %s:%d/%s ", d, devices[d]->name, port,
              portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");
          ncclIbDevs[ncclNIbDevs].device = d;
          ncclIbDevs[ncclNIbDevs].port = port;
          ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
          ncclIbDevs[ncclNIbDevs].context = context;
          strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
          readFileNumber(&vendorId, IB_DEVICE_SYSFS_FMT, devices[d]->name, "vendor");
          readFileNumber(&devId, IB_DEVICE_SYSFS_FMT, devices[d]->name, "device");
          ncclIbDevs[ncclNIbDevs].isSharpDev = 0;
          if ((portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) &&
              (vendorId == 0x15b3) &&           // Mellanox vendor
              (devId == 4123 || devId == 4124)) //ConnectX-6
          {
            ncclIbDevs[ncclNIbDevs].isSharpDev = 1;
            ncclNSharpDevs++;
          }
          ncclNIbDevs++;
          found++;
        }
        if (found == 0 && ncclSuccess != wrap_ibv_close_device(context)) { return ncclInternalError; }
      }
      if (nIbDevs && (ncclSuccess != wrap_ibv_free_device_list(devices))) { return ncclInternalError; };
    }
    if (ncclNIbDevs == 0) {
      INFO(NCCL_INIT|NCCL_NET, "NET/UCX : No device found.");
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
      INFO(NCCL_INIT|NCCL_NET, "NET/UCX : Using%s ; OOB %s:%s", line, if_name, socketToString(&nccl_ucx_if_addr.sa, addrline));
    }
    pthread_mutex_unlock(&nccl_ucx_lock);
  }
  return ncclSuccess;
}

ncclResult_t nccl_ucx_listen(int dev, void *handle, void **listen_comm) {
  ucx_listen_handle *my_handle;
  ucx_listen_comm *comm;

  comm = malloc(sizeof(ucx_listen_comm));
  memset(comm, 0, sizeof(ucx_listen_comm));
  NCCL_STATIC_ASSERT(sizeof(ucx_listen_handle) < NCCL_NET_HANDLE_MAXSIZE, "ucx listen handle size too large");
  my_handle = (ucx_listen_handle *)handle;
  comm->dev = dev;
  NCCLCHECK(get_socket_addr(&(my_handle->connectAddr)));
  NCCLCHECK(createListenSocket(&comm->fd, &my_handle->connectAddr));
  NCCLCHECK(ucx_get_ctx_and_worker(dev, &comm->ctx, &comm->worker, &comm->tag));
  my_handle->tag = comm->tag;
  *listen_comm = comm;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_connect(int dev, void *handle, void **send_comm) {
  ucp_address_t *my_addr;
  size_t local_addr_len;
  size_t rkey_buf_size;
  void *rkey_buf;

  ucx_listen_handle *recv_handle = (ucx_listen_handle *)handle;
  ucx_send_comm *comm = (ucx_send_comm *) calloc(1, sizeof(ucx_send_comm));
 
  NCCLCHECK(connectAddress(&comm->fd, &recv_handle->connectAddr));
  NCCLCHECK(ucx_get_ctx_and_worker(dev, &comm->ctx, &comm->worker, &comm->ctag));
  comm->tag = recv_handle->tag;
  comm->gpuFlush.enabled = 0;
  NCCLCHECK(ucx_worker_get_netaddress(comm->worker, &my_addr, &local_addr_len));
  INFO(NCCL_NET, "Worker address length: %zu", local_addr_len);

  ucp_mem_map_params_t mmap_params;
  uint64_t tail_adr = (uint64_t)&comm->fifo_tail;
  mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH;
  mmap_params.address = (void*)tail_adr;
  mmap_params.length = sizeof(uint32_t);
  ucp_mem_map(comm->ctx, &mmap_params, &comm->fifo_memh);
  ucp_rkey_pack(comm->ctx, comm->fifo_memh, &rkey_buf, &rkey_buf_size);

  NCCLCHECK(socketSend(comm->fd, &rkey_buf_size, sizeof(size_t)));
  NCCLCHECK(socketSend(comm->fd, rkey_buf, rkey_buf_size));
  NCCLCHECK(socketSend(comm->fd, &tail_adr, sizeof(uint64_t)));
  NCCLCHECK(socketSend(comm->fd, &local_addr_len, sizeof(size_t)));
  NCCLCHECK(socketSend(comm->fd, my_addr, local_addr_len));
  NCCLCHECK(socketSend(comm->fd, &comm->ctag, sizeof(ucp_tag_t)));
  *send_comm = comm;
  free(my_addr);
  free(rkey_buf);

  return ncclSuccess;
}

#define REG_ALIGN (4096)
ncclResult_t nccl_ucx_regmr(void* comm, void* data, int size, int type, void** mhandle) {
  ucp_mem_map_params_t mmap_params;
  ucx_ctx *ctx = (ucx_ctx*)comm;
  ucx_mhandle *mh;
  
  uint64_t addr = (uint64_t)data;
  uint64_t reg_addr = addr & (~(REG_ALIGN-1));
  uint64_t reg_size = addr+size - reg_addr;
  reg_size = ((reg_size + REG_ALIGN-1) / REG_ALIGN ) * REG_ALIGN;

  mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH; 
  mmap_params.address    = (void*)reg_addr;
  mmap_params.length     = reg_size;
  
  mh = (ucx_mhandle*)malloc(sizeof(ucx_mhandle));
  ucp_mem_map(ctx->ucp_ctx, &mmap_params, &mh->ucp_memh);
  if (ctx->gpuFlush.enabled) {
    size_t rkey_buf_size;
    void *rkey_buf;
    ucp_rkey_pack(ctx->ucp_ctx, mh->ucp_memh, &rkey_buf, &rkey_buf_size);
    UCXCHECK(ucp_ep_rkey_unpack(ctx->gpuFlush.flush_ep, rkey_buf, &mh->rkey));
  }
  
  *mhandle = mh;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_deregmr(void* comm, void* mhandle) {
  ucx_ctx *ctx = (ucx_ctx*)comm;
  ucx_mhandle *mh = (ucx_mhandle*)mhandle;
  if (ctx->gpuFlush.enabled) {
      ucp_rkey_destroy(mh->rkey);
  }
  ucp_mem_unmap(ctx->ucp_ctx, mh->ucp_memh);
  free(mhandle);
  return ncclSuccess;
}

ncclResult_t nccl_ucx_accept(void *listen_comm, void **recv_comm) {
  ucx_recv_comm *r_comm = (ucx_recv_comm *)calloc(1, sizeof(ucx_recv_comm));
  ucx_listen_comm *l_comm = (ucx_listen_comm *)listen_comm;
  void *rkey_buf;
  size_t rkey_buf_size;

  struct sockaddr_in sockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  SYSCHECKVAL(accept(l_comm->fd, (struct sockaddr*)&sockaddr, &socklen), "accept", r_comm->fd);

  r_comm->ctx = l_comm->ctx; r_comm->worker = l_comm->worker; r_comm->tag = l_comm->tag;
  NCCLCHECK(socketReceive(r_comm->fd, &rkey_buf_size, sizeof(size_t)));
  rkey_buf = malloc(rkey_buf_size);
  NCCLCHECK(socketReceive(r_comm->fd, rkey_buf, rkey_buf_size));
  NCCLCHECK(socketReceive(r_comm->fd, &r_comm->rem_tail_addr, sizeof(uint64_t)));

  size_t peer_addr_len;
  ucp_address_t *peer_addr;
  ucp_ep_params_t ep_params;
  NCCLCHECK(socketReceive(r_comm->fd, &peer_addr_len, sizeof(size_t)));
  peer_addr = malloc(peer_addr_len);
  NCCLCHECK(socketReceive(r_comm->fd, peer_addr, peer_addr_len));
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS; //|
  //                         UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
  ep_params.address = peer_addr;
  //  ep_params.err_mode        = err_handling_opt.ucp_err_mode;
  UCXCHECK(ucp_ep_create(r_comm->worker, &ep_params, &r_comm->ep));
  UCXCHECK(ucp_ep_rkey_unpack(r_comm->ep, rkey_buf, &r_comm->rkey));
  NCCLCHECK(socketReceive(r_comm->fd, &r_comm->ctag, sizeof(ucp_tag_t)));

  r_comm->gpuFlush.enabled = (nccl_ucx_gdr_support(l_comm->dev) == 0);  
  if (r_comm->gpuFlush.enabled) {
    ucp_address_t *my_addr;
    size_t local_addr_len;

    NCCLCHECK(ucx_worker_get_netaddress(r_comm->worker, &my_addr, &local_addr_len));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = my_addr;
    UCXCHECK(ucp_ep_create(r_comm->worker, &ep_params, &r_comm->gpuFlush.flush_ep));
  }
  free(peer_addr);
  free(rkey_buf);
  *recv_comm = r_comm;

  return ncclSuccess;
}

ncclResult_t ucx_send_check(ucx_send_comm *comm) {
  ucp_tag_message_h msg_tag;
  ucp_tag_recv_info_t info_tag;
  ucx_request *req;
  connect_msg *msg;
  ucp_ep_params_t ep_params;

  ucp_worker_progress(comm->worker);

  msg_tag = ucp_tag_probe_nb(comm->worker, comm->ctag, tag_mask, 1, &info_tag);
  if (msg_tag == NULL) {
    return ncclSuccess;
  }
  msg = malloc(info_tag.length);
  req = ucp_tag_msg_recv_nb(comm->worker, msg, info_tag.length, ucp_dt_make_contig(1), msg_tag, recv_handler);
  if (UCS_PTR_IS_ERR(req)) {
    WARN("Unable to receive connect msg (%s)", ucs_status_string(UCS_PTR_STATUS(req)));
  }
  else {
    while (req->completed == 0) {
      ucp_worker_progress(comm->worker);
    }
    req->completed = 0;
    ucp_request_release(req);
  }
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS; //|
  //                         UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
  ep_params.address = (ucp_address_t*)(msg + 1);
  //  ep_params.err_mode        = err_handling_opt.ucp_err_mode;
  UCXCHECK(ucp_ep_create(comm->worker, &ep_params, &comm->ep));
  comm->ready = 1;
  free(msg);

  return ncclSuccess;
}

ncclResult_t ucx_recv_check(ucx_recv_comm *comm) {
  if (comm->connect_req == NULL){
    ucp_address_t *my_addr;
    size_t local_addr_len;
    NCCLCHECK(ucx_worker_get_netaddress(comm->worker, &my_addr, &local_addr_len));
    size_t msg_len = sizeof(connect_msg) + local_addr_len;
    comm->msg = calloc(1, msg_len);
    comm->msg->addr_len = local_addr_len;
    memcpy(comm->msg + 1, my_addr, local_addr_len);
    comm->connect_req = ucp_tag_send_nb(comm->ep, comm->msg, msg_len, ucp_dt_make_contig(1), comm->ctag, send_handler);
    if (UCS_PTR_IS_ERR(comm->connect_req)) {
      WARN("Unable to send connect message");
      return ncclSystemError;
    } else if (comm->connect_req == NULL){
      comm->ready = 1;
      free(comm->msg);
    }
    free(my_addr);
  }
  else{
    if (comm->connect_req->completed == 1) {
      comm->ready = 1;
      comm->connect_req->completed = 0;
      ucp_request_release(comm->connect_req);
      free(comm->msg);
    }
    else {
      ucp_worker_progress(comm->worker);
    }
  }
  return ncclSuccess;
}

ncclResult_t nccl_ucx_isend(void *send_comm, void *data, int size, void *mhandle, void **request) {
  ucx_request *req;
  ucx_send_comm *comm = (ucx_send_comm *)send_comm;

  if (comm->ready == 0) { ucx_send_check(comm); }
  if (comm->ready == 0) { *request = NULL; return ncclSuccess; }

  volatile uint32_t *head = &comm->fifo_head;
  volatile uint32_t *tail = &comm->fifo_tail;
  if (*head == *tail) { *request = NULL; return ncclSuccess; }
  req = ucp_tag_send_nb(comm->ep, data, size, ucp_dt_make_contig(1), comm->tag, send_handler);
  if (UCS_PTR_IS_ERR(req)) {
    WARN("ucx_isend: unable to send message (%s)\n", ucs_status_string(UCS_PTR_STATUS(req)));
    return ncclSystemError;
  }
  else if (req != NULL) {
    ucp_worker_progress(comm->worker);
    req->worker = comm->worker;
    req->size = size;
  }
  comm->fifo_head++;
  *request = req ? req : (size ? 1: 2);

  return ncclSuccess;
}

ncclResult_t nccl_ucx_irecv(void *recv_comm, void *data, int size, void *mhandle, void **request) {
  ucx_request *req;
  ucx_recv_comm *comm = (ucx_recv_comm *)recv_comm;

  if (comm->ready == 0) { ucx_recv_check(comm); }
  if (comm->ready == 0) { *request = NULL; return ncclSuccess; }
  req = ucp_tag_recv_nb(comm->worker, data, size, ucp_dt_make_contig(1), comm->tag, tag_mask, recv_handler);
  if (UCS_PTR_IS_ERR(req)) {
    WARN("ucx_irecv: unable to receive message (%s)", ucs_status_string(UCS_PTR_STATUS(req)));
    return ncclSystemError;
  }
  else if (req != NULL) {
    ucp_worker_progress(comm->worker);
    req->worker = comm->worker;
    req->size = size;
  }
  comm->tail++;
  ucp_put_nbi(comm->ep, &comm->tail, sizeof(uint32_t), comm->rem_tail_addr, comm->rkey);
  *request = req ? req : (size ? 1: 2);

  return ncclSuccess;
}

ncclResult_t nccl_ucx_test(void *request, int *done, int *size) {
  ucx_request *req = (ucx_request *)request;
  *done = 0;
  if ((uint64_t)request == 1ul || (uint64_t)request == 2ul) {
    *done = 1;
    if (size) *size = -1 + (uint64_t)request;
    return ncclSuccess;
  }
  if (req->completed == 1) {
    *done = 1;
    if (size) *size = req->size;
    req->completed = 0;
  }
  else {
    ucp_worker_progress(req->worker);
  }

  return ncclSuccess;
}

ncclResult_t nccl_ucx_flush(void* recv_comm, void* data, int size, void* mhandle) {
  ucx_recv_comm *comm = (ucx_recv_comm *)recv_comm;
  if (comm->gpuFlush.enabled == 0 || size == 0) return ncclSuccess;
  ucx_mhandle *mh = (ucx_mhandle*)mhandle;
  ucx_request *req;
  req = ucp_get_nb(comm->gpuFlush.flush_ep, &comm->gpuFlush.hostMem, 1, (uint64_t)data, mh->rkey, send_handler);
  if (UCS_PTR_IS_ERR(req)) {
    WARN("ucx_flush: unable to read data (%s)", ucs_status_string(UCS_PTR_STATUS(req)));
    return ncclSystemError;
  } else if (req != NULL){
    while(req->completed == 0){
      ucp_worker_progress(comm->worker);
    }
    req->completed = 0;
    ucp_request_release(req);
  }
  return ncclSuccess;
}

static void wait_close(ucp_worker_h worker, ucx_request *req) {
  ucs_status_t status;

  if (UCS_PTR_IS_PTR(req)) {
    do {
      ucp_worker_progress(worker);
      status = ucp_request_check_status(req);
    } while(status == UCS_INPROGRESS);
    req->completed = 0;
    ucp_request_free(req);
  } else if (req != NULL) {
      WARN("Failed to close UCX endpoint");
  }
}

ncclResult_t nccl_ucx_close_send(void *send_comm) {
  if (send_comm){
    ucx_send_comm *comm = (ucx_send_comm*) send_comm;
    ucp_mem_unmap(comm->ctx, comm->fifo_memh);
    
    void *close_req;
    if (comm->ep) {
      close_req = ucp_ep_close_nb(comm->ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->worker, close_req);  
    }

    close(comm->fd);
    free(comm);
  }

  return ncclSuccess;
}

ncclResult_t nccl_ucx_close_recv(void *recv_comm) {
  if (recv_comm){
    ucx_recv_comm *comm = (ucx_recv_comm*)recv_comm;
    ucp_rkey_destroy(comm->rkey);

    void *close_req;
    if (comm->ep) {
      close_req = ucp_ep_close_nb(comm->ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->worker, close_req);  
    }
    if (comm->gpuFlush.enabled) {
      close_req = ucp_ep_close_nb(comm->gpuFlush.flush_ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->worker, close_req);
    }

    close(comm->fd);
    free(comm);
  }
  
  return ncclSuccess;
}

ncclResult_t nccl_ucx_close_listen(void *listen_comm) {
  ucx_listen_comm *comm = (ucx_listen_comm *)listen_comm;
  if (comm) {
    close(comm->fd);
    free(comm);
  }
  
  return ncclSuccess;
}

ncclNet_t ucx_plugin ={
    "UCX",
    nccl_ucx_init,
    nccl_ucx_devices,
    nccl_ucx_pci_path,
    nccl_ucx_ptr_support,
    nccl_ucx_listen,
    nccl_ucx_connect,
    nccl_ucx_accept,
    nccl_ucx_regmr,
    nccl_ucx_deregmr,
    nccl_ucx_isend,
    nccl_ucx_irecv,
    nccl_ucx_flush,
    nccl_ucx_test,
    nccl_ucx_close_send,
    nccl_ucx_close_recv,
    nccl_ucx_close_listen
};
