/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 * Copyright (C) 2019-2020, Mellanox Technologies Ltd. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <pthread.h>
#include <stdint.h>
#include <sys/time.h>
#include <unistd.h>

#include "core.h"
#include "ibvwrap.h"
#include "nccl.h"
#include "nccl_net.h"
#include "p2p_plugin.h"
#include "param.h"
#include "socket.h"
#include "ucp/api/ucp.h"

#define UCXCHECK(cmd) do {                           \
  int e = cmd;                                       \
  if( UCS_OK != e ) {                                \
    WARN("Failed: UCX error %s:%d '%d' %s\n",        \
        __FILE__,__LINE__, e, ucs_status_string(e)); \
    return ncclInternalError;                        \
  }                                                  \
} while(0)

#define UCXCHECK_VOID(cmd) do {                      \
  int e = cmd;                                       \
  if( UCS_OK != e ) {                                \
    WARN("Failed: UCX error %s:%d '%d' %s\n",        \
        __FILE__,__LINE__, e, ucs_status_string(e)); \
  }                                                  \
} while(0)

NCCL_PARAM(UCXDisable, "UCX_DISABLE", 0);

extern ncclDebugLogger_t pluginLogFunction;
static const ucp_tag_t tag      = 0xABADBABE;
static const ucp_tag_t tag_mask = 0xFFFFFFFFFFFFFFFF;

static int ncclNIbDevs = -1;
extern int ncclNSharpDevs;

typedef struct ucx_mhandle {
  ucp_mem_h  ucp_memh;
  ucp_rkey_h rkey;
} ucx_mhandle_t;

struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
struct userIbDev userIbDevs[MAX_IB_DEVS];

ncclResult_t nccl_ucx_devices(int* ndev) {
  *ndev = ncclNIbDevs;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_get_properties(int dev, ncclNetProperties_t* props)
{
  return nccl_p2p_ib_get_properties(ncclIbDevs, dev, props);
}

pthread_mutex_t nccl_ucx_lock = PTHREAD_MUTEX_INITIALIZER;

typedef struct ucx_listen_handle {
  union socketAddress connectAddr;
  ucp_tag_t           tag;
} ucx_listen_handle_t;

typedef struct ucx_listen_comm {
  int           dev;
  int           fd;
  ucp_context_h ctx;
  ucp_worker_h  worker;
  ucp_tag_t     tag;
} ucx_listen_comm_t;

typedef struct connect_msg {
  size_t addr_len;
} connect_msg_t;

typedef struct ucx_request {
  int          completed;
  int          size;
  ucp_worker_h worker;
} ucx_request_t;

struct ep_list {
  int            fd;
  struct ep_list *next;
};

struct nccl_ucx_worker {
  ucp_context_h  ctx;
  ucp_worker_h   worker;
  int            count;
  struct ep_list *eps;
  ucp_tag_t      last_tag;
};
static struct nccl_ucx_worker workers[MAX_IB_DEVS];

typedef struct ucx_gpu_flush {
  int      enabled;
  int      hostMem;
  ucp_ep_h flush_ep;
} ucx_gpu_flush_t;

typedef struct ucx_ctx {
  ucp_context_h   ucp_ctx;
  ucx_gpu_flush_t gpuFlush;
} ucx_ctx_t;

typedef struct ucx_send_comm {
  ucp_context_h   ctx;
  ucx_gpu_flush_t gpuFlush;
  ucp_worker_h    worker;
  ucp_ep_h        ep;
  ucp_tag_t       tag;
  ucp_tag_t       ctag;
  int             fd;
  int             ready;
  uint32_t        fifo_head;
  uint32_t        *fifo_tail;
  ucp_mem_h       fifo_memh;
} ucx_send_comm_t;

typedef struct ucx_recv_comm {
  ucp_context_h   ctx;
  ucx_gpu_flush_t gpuFlush;
  ucp_worker_h    worker;
  ucp_ep_h        ep;
  ucp_tag_t       tag;
  ucp_tag_t       ctag;
  int             fd;
  int             ready;
  uint64_t        rem_tail_addr;
  uint32_t        tail;
  ucp_rkey_h      rkey;
  connect_msg_t   *msg;
  ucx_request_t   *connect_req;
} ucx_recv_comm_t;

static void request_init(void *request) {
  ucx_request_t *req = (ucx_request_t*)request;
  req->completed = 0;
}

static void send_handler(void *request, ucs_status_t status) {
  ucx_request_t *req = (ucx_request_t*)request;
  req->completed = 1;
}

static void recv_handler(void *request, ucs_status_t status, ucp_tag_recv_info_t *info) {
  ucx_request_t *req = (ucx_request_t*)request;
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
  char         ucx_dev_name[PATH_MAX];

  snprintf(ucx_dev_name, PATH_MAX, "%s:%d", ncclIbDevs[dev].devName, ncclIbDevs[dev].port);
  UCXCHECK(ucp_config_read("NCCL", NULL, &config));
  UCXCHECK(ucp_config_modify(config, "NET_DEVICES", ucx_dev_name));

  memset(&ucp_params, 0, sizeof(ucp_params));
  ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES |
                            UCP_PARAM_FIELD_REQUEST_SIZE |
                            UCP_PARAM_FIELD_REQUEST_INIT;
  ucp_params.features     = UCP_FEATURE_TAG | UCP_FEATURE_RMA;
  ucp_params.request_size = sizeof(ucx_request_t);
  ucp_params.request_init = request_init;

  UCXCHECK(ucp_init(&ucp_params, config, ctx));
  ucp_config_release(config);

  return ncclSuccess;
}

static ncclResult_t ucx_init_worker(ucp_context_h ctx, ucp_worker_h *worker) {
  ucp_worker_params_t worker_params;

  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
  UCXCHECK(ucp_worker_create(ctx, &worker_params, worker));

  return ncclSuccess;
}

static ncclResult_t ucx_worker_get_netaddress(ucp_worker_h worker, ucp_address_t **address, size_t *address_length) {
  ucp_worker_attr_t attr;

  attr.field_mask    = UCP_WORKER_ATTR_FIELD_ADDRESS |
                       UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS;
  attr.address_flags = UCP_WORKER_ADDRESS_FLAG_NET_ONLY;

  UCXCHECK(ucp_worker_query(worker, &attr));
  *address = malloc(attr.address_length);
  if (address == NULL) {
    return ncclSystemError;
  }

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
    workers[dev].last_tag = tag;
  }

  *ctx    = workers[dev].ctx;
  *worker = workers[dev].worker;

  if (newtag != NULL) {
    workers[dev].last_tag += 1;
    *newtag = workers[dev].last_tag;
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

static ncclResult_t nccl_ucx_free_worker(ucp_worker_h worker) {
  int i;
  int dummy;
  struct ep_list *ep, *cur;

  pthread_mutex_lock(&nccl_ucx_lock);
  for(i = 0; i < ncclNIbDevs; i++) {
    if (worker == workers[i].worker) {
      workers[i].count--;
      if (workers[i].count == 0){
        ep = workers[i].eps;
        while(ep){
          cur = ep;
          NCCLCHECK(socketReceive(ep->fd, &dummy, sizeof(int)));
          ep = ep->next;
          close(cur->fd);
          free(cur);
        }
        ucp_worker_destroy(workers[i].worker);
        ucp_cleanup(workers[i].ctx);
        workers[i].eps    = NULL;
        workers[i].worker = NULL;
        workers[i].ctx    = NULL;
      }
      break;
    }
  }
  pthread_mutex_unlock(&nccl_ucx_lock);

  return ncclSuccess;
}

static ncclResult_t nccl_ucx_add_ep(ucp_worker_h worker, int fd) {
  ncclResult_t status = ncclSuccess;
  int i;

  for(i = 0; i < ncclNIbDevs; i++) {
    if (worker == workers[i].worker) {
      struct ep_list *new_ep = (struct ep_list*)malloc(sizeof(struct ep_list));

      if (new_ep == NULL) {
        status = ncclSystemError;
        break;
      }

      new_ep->fd   = fd;
      new_ep->next = workers[i].eps;
      workers[i].eps = new_ep;
      break;
    }
  }

  return status;
}

ncclResult_t nccl_ucx_init(ncclDebugLogger_t logFunction) {
  if (ncclParamUCXDisable()) return ncclInternalError;

  return nccl_p2p_ib_init(&ncclNIbDevs, ncclIbDevs, if_name, &nccl_ucx_if_addr, NULL, logFunction);
}

ncclResult_t nccl_ucx_listen(int dev, void *handle, void **listen_comm) {
  ucx_listen_handle_t *my_handle = (ucx_listen_handle_t*)handle;
  ucx_listen_comm_t   *comm      = (ucx_listen_comm_t*)calloc(1, sizeof(*comm));

  NCCL_STATIC_ASSERT(sizeof(ucx_listen_handle_t) < NCCL_NET_HANDLE_MAXSIZE, "UCX listen handle size too large");
  NCCLCHECK(get_socket_addr(&(my_handle->connectAddr)));
  NCCLCHECK(createListenSocket(&comm->fd, &my_handle->connectAddr));
  NCCLCHECK(ucx_get_ctx_and_worker(dev, &comm->ctx, &comm->worker, &comm->tag));

  comm->dev = dev;
  my_handle->tag = comm->tag;
 
  *listen_comm = comm;
 
  return ncclSuccess;
}

ncclResult_t nccl_ucx_connect(int dev, void *handle, void **send_comm) {
  ucx_listen_handle_t  *recv_handle = (ucx_listen_handle_t*)handle;
  ucx_send_comm_t      *comm;
  ucp_address_t        *my_addr;
  ucp_mem_map_params_t mmap_params;
  size_t               local_addr_len;
  size_t               rkey_buf_size;
  void                 *rkey_buf;
  uint64_t             tail_adr;

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(ucx_send_comm_t)));
  memset(comm, 0, sizeof(ucx_send_comm_t));
  NCCLCHECK(connectAddress(&comm->fd, &recv_handle->connectAddr));
  NCCLCHECK(ucx_get_ctx_and_worker(dev, &comm->ctx, &comm->worker, &comm->ctag));
  comm->tag              = recv_handle->tag;
  comm->gpuFlush.enabled = 0;
  NCCLCHECK(ucx_worker_get_netaddress(comm->worker, &my_addr, &local_addr_len));
  nccl_ucx_add_ep(comm->worker,comm->fd);
  INFO(NCCL_NET, "Worker address length: %zu", local_addr_len);

  NCCLCHECK(ncclIbMalloc((void**)&comm->fifo_tail, sizeof(uint32_t)));
  tail_adr = (uint64_t)comm->fifo_tail;
  mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH;
  mmap_params.address    = (void*)tail_adr;
  mmap_params.length     = sizeof(uint32_t);
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

ncclResult_t nccl_ucx_accept(void *listen_comm, void **recv_comm) {
  ucx_listen_comm_t  *l_comm = (ucx_listen_comm_t *)listen_comm;
  socklen_t          socklen = sizeof(struct sockaddr_in);
  ucx_recv_comm_t    *r_comm;
  void   *           rkey_buf;
  size_t             rkey_buf_size;
  size_t             peer_addr_len;
  ucp_address_t      *peer_addr;
  ucp_ep_params_t    ep_params;
  struct sockaddr_in sockaddr;

  NCCLCHECK(ncclIbMalloc((void**)&r_comm, sizeof(ucx_recv_comm_t)));
  memset(r_comm, 0, sizeof(ucx_recv_comm_t));

  SYSCHECKVAL(accept(l_comm->fd, (struct sockaddr*)&sockaddr, &socklen), "accept", r_comm->fd);

  r_comm->ctx    = l_comm->ctx;
  r_comm->worker = l_comm->worker;
  r_comm->tag    = l_comm->tag;

  NCCLCHECK(socketReceive(r_comm->fd, &rkey_buf_size, sizeof(size_t)));

  rkey_buf = malloc(rkey_buf_size);
  if (rkey_buf == NULL) {
    return ncclSystemError;
  }

  NCCLCHECK(socketReceive(r_comm->fd, rkey_buf, rkey_buf_size));
  NCCLCHECK(socketReceive(r_comm->fd, &r_comm->rem_tail_addr, sizeof(uint64_t)));
  NCCLCHECK(socketReceive(r_comm->fd, &peer_addr_len, sizeof(size_t)));

  peer_addr = malloc(peer_addr_len);
  if (peer_addr == NULL) {
    free(rkey_buf);
    return ncclSystemError;
  }

  NCCLCHECK(socketReceive(r_comm->fd, peer_addr, peer_addr_len));
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS; //|
  //                         UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
  ep_params.address = peer_addr;
  UCXCHECK(ucp_ep_create(r_comm->worker, &ep_params, &r_comm->ep));
  UCXCHECK(ucp_ep_rkey_unpack(r_comm->ep, rkey_buf, &r_comm->rkey));
  NCCLCHECK(socketReceive(r_comm->fd, &r_comm->ctag, sizeof(ucp_tag_t)));

  r_comm->gpuFlush.enabled = (nccl_p2p_gdr_support(l_comm->dev) == ncclSuccess);  
  if (r_comm->gpuFlush.enabled) {
    ucp_address_t *my_addr;
    size_t        local_addr_len;

    NCCLCHECK(ucx_worker_get_netaddress(r_comm->worker, &my_addr, &local_addr_len));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address    = my_addr;
    UCXCHECK(ucp_ep_create(r_comm->worker, &ep_params, &r_comm->gpuFlush.flush_ep));
    free(my_addr);
  }

  free(peer_addr);
  free(rkey_buf);
  *recv_comm = r_comm;

  return ncclSuccess;
}

ncclResult_t nccl_ucx_regmr(void* comm, void* data, int size, int type, void** mhandle) {
  WARN("NET/UCX: not implemented");
  return ncclInternalError;
}

ncclResult_t nccl_ucx_deregmr(void* comm, void* mhandle) {
  WARN("NET/UCX: not implemented");
  return ncclInternalError;
}

ncclResult_t nccl_ucx_isend(void *send_comm, void *data, int size, void *mhandle, void **request) {
  WARN("NET/UCX: not implemented");
  return ncclInternalError;
}

ncclResult_t nccl_ucx_irecv(void *recv_comm, void *data, int size, void *mhandle, void **request) {
  WARN("NET/UCX: not implemented");
  return ncclInternalError;
}

ncclResult_t nccl_ucx_flush(void* recv_comm, void* data, int size, void* mhandle) {
  WARN("NET/UCX: not implemented");
  return ncclInternalError;
}

ncclResult_t nccl_ucx_test(void *request, int *done, int *size) {
  WARN("NET/UCX: not implemented");
  return ncclInternalError;
}

static void wait_close(ucp_worker_h worker, ucx_request_t *req) {
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
  void *close_req;

  if (send_comm){
    ucx_send_comm_t *comm = (ucx_send_comm_t*) send_comm;
    ucp_mem_unmap(comm->ctx, comm->fifo_memh);
    
    if (comm->ep) {
      close_req = ucp_ep_close_nb(comm->ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->worker, close_req);
      int close = 1;
      NCCLCHECK(socketSend(comm->fd, &close, sizeof(int)));
    }
    nccl_ucx_free_worker(comm->worker);
    free(comm->fifo_tail);
    free(comm);
  }

  return ncclSuccess;
}

ncclResult_t nccl_ucx_close_recv(void *recv_comm) {
  void *close_req;

  if (recv_comm){
    ucx_recv_comm_t *comm = (ucx_recv_comm_t*)recv_comm;
    ucp_rkey_destroy(comm->rkey);

    if (comm->gpuFlush.enabled) {
      close_req = ucp_ep_close_nb(comm->gpuFlush.flush_ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->worker, close_req);
    }
    if (comm->ep) {
      close_req = ucp_ep_close_nb(comm->ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->worker, close_req);
      int close=1;
      NCCLCHECK(socketSend(comm->fd, &close, sizeof(int)));  
    }
    nccl_ucx_free_worker(comm->worker);
    free(comm);
  }
  
  return ncclSuccess;
}

ncclResult_t nccl_ucx_close_listen(void *listen_comm) {
  ucx_listen_comm_t *comm = (ucx_listen_comm_t *)listen_comm;

  if (comm) {
    close(comm->fd);
    free(comm);
  }
  
  return ncclSuccess;
}

ncclNet_t ucxPlugin = {
  "UCX",
  nccl_ucx_init,
  nccl_ucx_devices,
  nccl_ucx_get_properties,
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
