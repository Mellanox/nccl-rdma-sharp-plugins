/*************************************************************************
 *  * Copyright (c) 2016-2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *   *
 *    * See LICENSE.txt for license information
 *     ************************************************************************/

#include <pthread.h>
#include <stdint.h>
#include <sys/time.h>
#include <unistd.h>

#include "core.h"
#include "ibvwrap.h"
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

NCCL_PARAM(UCXRMADisable, "UCX_RMA_DISABLE", 0);

extern ncclDebugLogger_t pluginLogFunction;
static char nccl_ucx_rma_tls[32] = "";
static char nccl_ucx_rma_zcopy_thresh[32] ="";
static int ncclNIbDevs = -1;

#define MAX_UCX_RKEY_BUF_SIZE 128
typedef struct nccl_ucx_rma_rkey_buf {
    int    index;
    int    id;
    char   buf[MAX_UCX_RKEY_BUF_SIZE];
    size_t rkey_buf_size;
    int    send;
} nccl_ucx_rma_rkey_buf_t;

enum ncclUCXCommState {
  ncclUCXCommStateStart = 0,
  ncclUCXCommStateConnect = 1,
  ncclUCXCommStateAccept = 3,
};

struct ncclUCXCommStage {
  enum ncclUCXCommState state;
  uint8_t iteration;
  void* sock;
  void* comm;
};

typedef struct ucx_rma_mhandle {
  ucp_mem_h               ucp_memh;
  ucp_rkey_h              rkey;
  nccl_ucx_rma_rkey_buf_t rkey_buf;
  int                     mem_type;
} ucx_rma_mhandle_t;

ncclResult_t nccl_ucx_rma_devices(int* ndev) {
  *ndev = ncclNIbDevs;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_get_properties(int dev, ncclNetProperties_t* props)
{
  return nccl_p2p_ib_get_properties(ncclIbDevs, dev, props);
}

ncclResult_t nccl_ucx_rma_get_properties_v6(int dev, ncclNetProperties_v6_t* props_v6)
{
  ncclNetProperties_t props;
  ncclResult_t ret = nccl_ucx_rma_get_properties(dev, &props);
  if (ret != ncclSuccess) return ret;
  props_v6->name = props.name;
  props_v6->pciPath = props.pciPath;
  props_v6->guid = props.guid;
  props_v6->ptrSupport = props.ptrSupport;
  props_v6->speed = props.speed;
  props_v6->latency = props.latency;
  props_v6->port = props.port;
  props_v6->maxComms = props.maxComms;
  props_v6->maxRecvs = props.maxRecvs;

  return ncclSuccess;
}

pthread_mutex_t nccl_ucx_rma_lock = PTHREAD_MUTEX_INITIALIZER;

typedef struct ucx_rma_listen_handle {
  union ncclSocketAddress connectAddr; /* reciever socket address */
  uint64_t magic;                      /* random number to help debugging */
  ucp_tag_t               tag;         /* tag that is used to distiguish data that was sent to
                                          this reciever. Required when shared worker is used. */
  struct ncclUCXCommStage stage;
} ucx_rma_listen_handle_t;

typedef struct nccl_ucx_rma_listen_comm {
  int dev;
  struct ncclSocket sock;/* socket for OOB connection */
  struct ncclUCXCommStage stage;
} nccl_ucx_rma_listen_comm_t;

struct ep_list {
  struct ncclSocket *sock;
  struct ep_list *next;
};

struct nccl_ucx_worker {
  ucp_context_h  ctx;
  ucp_worker_h   worker;
  int            count;
  struct ep_list *eps;
};
static struct nccl_ucx_worker workers[MAX_IB_DEVS];

typedef struct ucx_gpu_flush {
  int      enabled;
  int      hostMem;
  ucp_ep_h flush_ep;
} ucx_gpu_flush_t;

enum {
  UCX_RMA_REQ_TYPE_SEND,
  UCX_RMA_REQ_TYPE_RECV,
  UCX_RMA_REQ_TYPE_FLUSH,
};

#define MAX_UCX_REQ_SIZE 256
typedef struct nccl_ucx_rma_request {
  char             ucx_req[MAX_UCX_REQ_SIZE];
  int              used;
  int              type;
  int              done;
  int              size;
  int              free;
  uint64_t         am_msg;
  int              seq;
  ucs_status_ptr_t st;
  ucp_worker_h     worker;
} nccl_ucx_rma_request_t;

typedef struct ucx_rma_send_fifo {
  uint64_t addr;
  uint64_t addr_request;
  int      size;
  uint32_t seq;
  uint32_t ready;
  int      rkey_idx;
  int      rkey_id;
  int      req_id;
} ucx_rma_send_fifo_t;

#define NCCL_UCX_RMA_MAX_MHANDLES 16
typedef struct nccl_ucx_rma_ctx {
  int                    id;
  int                    ready;
  struct ncclSocket	 sock;
  ucs_status_ptr_t       check_req;
  ucp_context_h          ctx;
  ucp_worker_h           worker;
  ucx_gpu_flush_t        gpuFlush;
  uint64_t               num_mh;
  ucx_rma_mhandle_t      *mh[NCCL_UCX_RMA_MAX_MHANDLES];
  nccl_ucx_rma_request_t reqs[MAX_REQUESTS];
} nccl_ucx_rma_ctx_t;

typedef struct nccl_ucx_rma_rkey {
  ucp_rkey_h rkey;
  int        id;
} nccl_ucx_rma_rkey_t;

typedef struct nccl_ucx_rma_send_comm {
  nccl_ucx_rma_ctx_t  super;
  ucp_ep_h            ep;
  ucx_rma_send_fifo_t fifo[MAX_REQUESTS];
  uint32_t            fifo_head;
  ucp_mem_h           fifo_memh;
  nccl_ucx_rma_rkey_t rkeys[NCCL_UCX_RMA_MAX_MHANDLES];
  int                 rem_am_id;
} nccl_ucx_rma_send_comm_t;

typedef struct ucx_rma_rem_fifo {
  ucx_rma_send_fifo_t elems[MAX_REQUESTS];
  uint64_t            addr;
  ucp_rkey_h          rkey;
  uint32_t            tail;
} ucx_rma_rem_fifo_t;

typedef struct nccl_ucx_rma_recv_comm {
  nccl_ucx_rma_ctx_t super;
  ucp_ep_h           ep;
  ucx_rma_rem_fifo_t rem_fifo;
  int                rem_am_id;
  void               *rkey_bufs;
} nccl_ucx_rma_recv_comm_t;


static union ncclSocketAddress nccl_ucx_if_addr;
static char if_name[MAX_IF_NAME_SIZE];

static ncclResult_t GetSocketAddr(union ncclSocketAddress *addr) {
  memcpy(addr, &nccl_ucx_if_addr, sizeof(*addr));
  return ncclSuccess;
}

typedef struct nccl_ucx_am_request {
  nccl_ucx_rma_request_t *req;
} nccl_ucx_am_request_t;

typedef nccl_ucx_am_request_t nccl_ucx_flush_request_t;

static ncclResult_t nccl_ucx_rma_init_ucp(int dev, ucp_context_h *ctx)
{
  ucp_params_t ucp_params;
  ucp_config_t *config;
  char         ucx_dev_name[PATH_MAX];

  snprintf(ucx_dev_name, PATH_MAX, "%s:%d", ncclIbDevs[dev].devName,
           ncclIbDevs[dev].port);
  UCXCHECK(ucp_config_read("NCCL", NULL, &config));

  UCXCHECK(ucp_config_modify(config, "NET_DEVICES", ucx_dev_name));
  UCXCHECK(ucp_config_modify(config, "TLS", nccl_ucx_rma_tls));
  UCXCHECK(ucp_config_modify(config, "ZCOPY_THRESH", nccl_ucx_rma_zcopy_thresh));

  memset(&ucp_params, 0, sizeof(ucp_params));
  ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES |
                            UCP_PARAM_FIELD_REQUEST_SIZE;
  ucp_params.features     = UCP_FEATURE_RMA |
                            UCP_FEATURE_AM;
  ucp_params.request_size = sizeof(nccl_ucx_am_request_t);

  UCXCHECK(ucp_init(&ucp_params, config, ctx));
  ucp_config_release(config);

  return ncclSuccess;
}

static ncclResult_t nccl_ucx_rma_init_worker(ucp_context_h ctx,
                                             ucp_worker_h *worker)
{
  ucp_worker_params_t worker_params;
  ucp_worker_attr_t   worker_attr;

  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_MULTI;

  UCXCHECK(ucp_worker_create(ctx, &worker_params, worker));

  worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE;
  ucp_worker_query(*worker, &worker_attr);
  if (worker_attr.thread_mode != UCS_THREAD_MODE_MULTI) {
    INFO(NCCL_NET, "Thread mode multi is not supported");
  }

  return ncclSuccess;
}

#define UCX_RMA_USE_SHARED_WORKER
static ncclResult_t nccl_ucx_rma_init_comm_context(int dev,
                                                   nccl_ucx_rma_ctx_t *comm_ctx)
{
  pthread_mutex_lock(&nccl_ucx_rma_lock);
#ifdef UCX_RMA_USE_SHARED_WORKER
  if (workers[dev].count == 0) {
    nccl_ucx_rma_init_ucp(dev, &workers[dev].ctx);
    nccl_ucx_rma_init_worker(workers[dev].ctx, &workers[dev].worker);
    workers->count = 0;
    workers->eps   = NULL;
  }

  comm_ctx->ctx    = workers[dev].ctx;
  comm_ctx->worker = workers[dev].worker;
  comm_ctx->id     = workers[dev].count;
  workers[dev].count++;
#else
  nccl_ucx_rma_init_ucp(dev, &comm_ctx->ctx);
  nccl_ucx_rma_init_worker(comm_ctx->ctx, &comm_ctx->worker);
#endif
  pthread_mutex_unlock(&nccl_ucx_rma_lock);
  return ncclSuccess;
}

static ncclResult_t nccl_ucx_rma_send_worker_address(ucp_worker_h worker, struct ncclSocket *sock)
{
  ucp_worker_attr_t attr;

  attr.field_mask    = UCP_WORKER_ATTR_FIELD_ADDRESS |
                       UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS;
  attr.address_flags = UCP_WORKER_ADDRESS_FLAG_NET_ONLY;

  UCXCHECK(ucp_worker_query(worker, &attr));
  NCCLCHECK(ncclSocketSend(sock, &attr.address_length, sizeof(attr.address_length)));
  NCCLCHECK(ncclSocketSend(sock, attr.address, attr.address_length));

  free(attr.address);
  return ncclSuccess;
}

static ncclResult_t nccl_ucx_free_worker(ucp_worker_h worker)
{
  int i;
  int dummy;
  struct ep_list *ep, *cur;

  pthread_mutex_lock(&nccl_ucx_rma_lock);
  for(i = 0; i < ncclNIbDevs; i++) {
    if (worker == workers[i].worker) {
      workers[i].count--;
      if (workers[i].count == 0) {
        ep = workers[i].eps;
        while(ep) {
          cur = ep;
          NCCLCHECK(ncclSocketRecv(ep->sock, &dummy, sizeof(int)));
          ep = ep->next;
          close(cur->sock->fd);
          free(cur);
        }
        ucp_worker_destroy(workers[i].worker);
        ucp_cleanup(workers[i].ctx);
        INFO(NCCL_NET, "worker destroy");
        workers[i].eps    = NULL;
        workers[i].worker = NULL;
        workers[i].ctx    = NULL;
      }
      break;
    }
  }
  pthread_mutex_unlock(&nccl_ucx_rma_lock);

  return ncclSuccess;
}

static ncclResult_t nccl_ucx_add_ep(ucp_worker_h worker, struct ncclSocket *sock)
{
  ncclResult_t status = ncclSuccess;
  int i;

  for(i = 0; i < ncclNIbDevs; i++) {
    if (worker == workers[i].worker) {
      struct ep_list *new_ep = (struct ep_list*)malloc(sizeof(struct ep_list));

      if (new_ep == NULL) {
        status = ncclSystemError;
        break;
      }

      new_ep->sock   = sock;
      new_ep->next = workers[i].eps;
      workers[i].eps = new_ep;
      break;
    }
  }

  return status;
}

ncclResult_t nccl_ucx_rma_init(ncclDebugLogger_t logFunction)
{
  char *config_env;
  if (ncclParamUCXRMADisable()) return ncclInternalError;
  NCCLCHECK(nccl_p2p_ib_init(&ncclNIbDevs, ncclIbDevs, if_name, &nccl_ucx_if_addr,
                          NULL, logFunction));

  if (strlen(nccl_ucx_rma_tls) == 0) {
    config_env = getenv("NCCL_UCX_TLS");
    if (config_env != NULL) {
      snprintf(nccl_ucx_rma_tls, 32, "%s", config_env);
    } else {
      snprintf(nccl_ucx_rma_tls, 32, "%s", "ib");
    }
    INFO(NCCL_NET, "NET/UCX_RMA: using transports: %s", nccl_ucx_rma_tls);
  }

  if (strlen(nccl_ucx_rma_zcopy_thresh) == 0) {
    config_env = getenv("NCCL_UCX_ZCOPY_THRESH");
    if (config_env != NULL) {
      snprintf(nccl_ucx_rma_zcopy_thresh, 32, "%s", config_env);
    } else {
      snprintf(nccl_ucx_rma_zcopy_thresh, 32, "%s", "1");
    }
    INFO(NCCL_NET, "NET/UCX_RMA: zero copy threshold: %s", nccl_ucx_rma_zcopy_thresh);
  }

  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_listen(int dev, void *handle, void **listen_comm)
{
  ucx_rma_listen_handle_t *my_handle = (ucx_rma_listen_handle_t*)handle;
  nccl_ucx_rma_listen_comm_t   *comm;

  NCCL_STATIC_ASSERT(sizeof(ucx_rma_listen_handle_t) < NCCL_NET_HANDLE_MAXSIZE,
                     "UCX-RMA listen handle size too large");

  my_handle->magic = NCCL_SOCKET_MAGIC;
  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(nccl_ucx_rma_listen_comm_t)));
  NCCLCHECK(ncclSocketInit(&comm->sock, &nccl_ucx_if_addr, my_handle->magic, ncclSocketTypeNetIb, NULL, 1));
  NCCLCHECK(ncclSocketListen(&comm->sock));
  NCCLCHECK(ncclSocketGetAddr(&comm->sock, &my_handle->connectAddr));

  comm->dev = dev; 
  *listen_comm = comm;
 
  return ncclSuccess;
}

static ucs_status_t nccl_ucx_rma_am_rkey_cb(void *arg, void *data, size_t length,
                                            ucp_ep_h reply_ep, unsigned flags)
{
  nccl_ucx_rma_send_comm_t *comm     = (nccl_ucx_rma_send_comm_t*)arg;
  nccl_ucx_rma_rkey_buf_t  *rkey_buf = (nccl_ucx_rma_rkey_buf_t*)data;
  
  if (comm->rkeys[rkey_buf->index].rkey) {
    ucp_rkey_destroy(comm->rkeys[rkey_buf->index].rkey);
  }
  comm->rkeys[rkey_buf->index].id = rkey_buf->id;
  UCXCHECK(ucp_ep_rkey_unpack(comm->ep, rkey_buf->buf,
                              &comm->rkeys[rkey_buf->index].rkey));
  return UCS_OK;
}


ncclResult_t nccl_ucx_rma_connect(int dev, void *handle, void **send_comm, ncclNetDeviceHandle_t** sendDevComm)
{
  ucx_rma_listen_handle_t  *recv_handle = (ucx_rma_listen_handle_t*)handle;
  struct ncclUCXCommStage* stage = &recv_handle->stage;
  nccl_ucx_rma_send_comm_t *comm;
  ucp_mem_map_params_t     mmap_params;
  size_t                   rkey_buf_size;
  void                     *rkey_buf;
  uint64_t                 fifo_adr;
  int                      i;
  int                      ready;

  *send_comm = NULL;

  if (stage->state == ncclUCXCommStateConnect) goto ucx_connect_check;

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(*comm)));
  NCCLCHECK(ncclSocketInit(&comm->super.sock, &recv_handle->connectAddr, recv_handle->magic, ncclSocketTypeNetIb, NULL, 1));
  stage->comm = comm;
  stage->state = ncclUCXCommStateConnect;
  NCCLCHECK(ncclSocketConnect(&comm->super.sock));

ucx_connect_check:
  /* since ncclSocketConnect is async, we must check if connection is complete */
  NCCLCHECK(ncclSocketReady(&comm->super.sock, &ready));
  if (!ready) return ncclSuccess;

  NCCLCHECK(nccl_ucx_rma_init_comm_context(dev, &comm->super));
  NCCLCHECK(nccl_ucx_rma_send_worker_address(comm->super.worker, &comm->super.sock));
  NCCLCHECK(nccl_ucx_add_ep(comm->super.worker, &comm->super.sock));
  UCXCHECK(ucp_worker_set_am_handler(comm->super.worker, comm->super.id,
                                     nccl_ucx_rma_am_rkey_cb, comm,
                                     UCP_AM_FLAG_WHOLE_MSG));
  for (i = 0; i < NCCL_UCX_RMA_MAX_MHANDLES; i++) {
    comm->rkeys[i].id = -1;
  }
  fifo_adr = (uint64_t)comm->fifo;
  mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH;
  mmap_params.address    = (void*)fifo_adr;
  mmap_params.length     = sizeof(ucx_rma_send_fifo_t) *
                           MAX_REQUESTS;
  UCXCHECK(ucp_mem_map(comm->super.ctx, &mmap_params, &comm->fifo_memh));
  UCXCHECK(ucp_rkey_pack(comm->super.ctx, comm->fifo_memh, &rkey_buf, &rkey_buf_size));
  NCCLCHECK(ncclSocketSend(&comm->super.sock, &rkey_buf_size, sizeof(size_t)));
  NCCLCHECK(ncclSocketSend(&comm->super.sock, rkey_buf, rkey_buf_size));
  NCCLCHECK(ncclSocketSend(&comm->super.sock, &fifo_adr, sizeof(uint64_t)));
  NCCLCHECK(ncclSocketSend(&comm->super.sock, &comm->super.id, sizeof(comm->super.id)));
  ucp_rkey_buffer_release(rkey_buf);
  *send_comm = comm;

  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_connect_v6(int dev, void *handle, void **send_comm)
{
  ncclNetDeviceHandle_v7_t* dev_handle = NULL;
  return nccl_ucx_rma_connect(dev, handle, send_comm, &dev_handle);
}

enum {
  NCCL_UCX_RMA_REQUEST_INPROGRESS = 0,
  NCCL_UCX_RMA_REQUEST_PUT_DONE   = 1,
  NCCL_UCX_RMA_REQUEST_AM_DONE    = 2,
  NCCL_UCX_RMA_REQUEST_DONE       = 3,
};

static ucs_status_t nccl_ucx_rma_am_cb(void *arg, void *data, size_t length,
                                       ucp_ep_h reply_ep, unsigned flags)
{
  nccl_ucx_rma_request_t *reqs = (nccl_ucx_rma_request_t*)arg;
  uint64_t *header = data;
  int      size    = *header & 0xFFFFFFFFFFFFFFFF;
  int      id      = *header >>32 ;

  reqs[id].size = size;
  reqs[id].done = NCCL_UCX_RMA_REQUEST_DONE;

  return UCS_OK;
}

static ncclResult_t nccl_ucx_rma_init_ep(struct ncclSocket *sock, ucp_worker_h worker, ucp_ep_h *ep, int blocking)
{
  int bytes = 0;
  ucp_ep_params_t ep_params;
  size_t          peer_addr_len;
  void            *peer_addr;

  if (blocking) {
    NCCLCHECK(ncclSocketRecv(sock, &peer_addr_len, sizeof(size_t)));
  } else {
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, sock, &peer_addr_len,
                            sizeof(size_t), &bytes));
    if (bytes == 0) {
      ep = NULL;
      return ncclSuccess;
    }
    NCCLCHECK(ncclSocketWait(NCCL_SOCKET_RECV, sock, &peer_addr_len,
                         sizeof(size_t), &bytes));
  }
  peer_addr = alloca(peer_addr_len);
  NCCLCHECK(ncclSocketRecv(sock, peer_addr, peer_addr_len));

  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
  ep_params.address    = peer_addr;
  UCXCHECK(ucp_ep_create(worker, &ep_params, ep));

  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_accept(void *listen_comm, void **recv_comm, ncclNetDeviceHandle_v7_t** recvDevComm)
{
  nccl_ucx_rma_listen_comm_t *l_comm = (nccl_ucx_rma_listen_comm_t *)listen_comm;
  socklen_t                  socklen = sizeof(struct sockaddr_in);
  struct ncclUCXCommStage* stage = &l_comm->stage;
  nccl_ucx_rma_recv_comm_t   *r_comm;
  struct sockaddr_in         sockaddr;
  void                       *rkey_buf;
  size_t                     rkey_buf_size;
  int                        ready;
 
  *recv_comm = NULL;
  if (stage->state == ncclUCXCommStateAccept) goto ucx_accept_check;

  NCCLCHECK(ncclIbMalloc((void**)&r_comm, sizeof(nccl_ucx_rma_recv_comm_t)));
  stage->comm = r_comm;
  stage->state = ncclUCXCommStateAccept;
  l_comm->sock.asyncFlag = 1;
  r_comm->super.sock.asyncFlag = 1;

  NCCLCHECK(ncclSocketInit(&r_comm->super.sock, NULL, NCCL_SOCKET_MAGIC, ncclSocketTypeUnknown, NULL, 0));
  NCCLCHECK(ncclSocketAccept(&r_comm->super.sock, &l_comm->sock));

ucx_accept_check:
  NCCLCHECK(ncclSocketReady(&r_comm->super.sock, &ready));
  if (!ready) return ncclSuccess;

  NCCLCHECK(nccl_ucx_rma_init_comm_context(l_comm->dev, &r_comm->super));
  UCXCHECK(ucp_worker_set_am_handler(r_comm->super.worker, r_comm->super.id,
                                     nccl_ucx_rma_am_cb, r_comm->super.reqs,
                                     UCP_AM_FLAG_WHOLE_MSG));

  NCCLCHECK(nccl_ucx_rma_init_ep(&r_comm->super.sock, r_comm->super.worker, &r_comm->ep, 1));
  NCCLCHECK(nccl_ucx_add_ep(r_comm->super.worker, &r_comm->super.sock));
  NCCLCHECK(ncclSocketRecv(&r_comm->super.sock, &rkey_buf_size, sizeof(size_t)));

  rkey_buf = malloc(rkey_buf_size);
  if (rkey_buf == NULL) {
    return ncclSystemError;
  }
  NCCLCHECK(ncclSocketRecv(&r_comm->super.sock, rkey_buf, rkey_buf_size));
  NCCLCHECK(ncclSocketRecv(&r_comm->super.sock, &r_comm->rem_fifo.addr, sizeof(uint64_t)));
  NCCLCHECK(ncclSocketRecv(&r_comm->super.sock, &r_comm->rem_am_id, sizeof(int)));
  UCXCHECK(ucp_ep_rkey_unpack(r_comm->ep, rkey_buf, &r_comm->rem_fifo.rkey));
  free(rkey_buf);

  if (nccl_p2p_gdr_support(l_comm->dev) == ncclSuccess) {
    r_comm->super.gpuFlush.enabled = 1;
  }

  if (r_comm->super.gpuFlush.enabled) {
    ucp_worker_attr_t attr;
    ucp_ep_params_t   ep_params;

    attr.field_mask    = UCP_WORKER_ATTR_FIELD_ADDRESS |
                         UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS;
    attr.address_flags = UCP_WORKER_ADDRESS_FLAG_NET_ONLY;

    UCXCHECK(ucp_worker_query(r_comm->super.worker, &attr));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address    = attr.address;
    UCXCHECK(ucp_ep_create(r_comm->super.worker, &ep_params,
                           &r_comm->super.gpuFlush.flush_ep));

    free(attr.address);
  }
  r_comm->super.num_mh = 0;
  *recv_comm = r_comm;

  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_accept_v6(void *listen_comm, void **recv_comm)
{
  ncclNetDeviceHandle_v7_t* dev_handle = NULL;
  return nccl_ucx_rma_accept(listen_comm, recv_comm, &dev_handle);
}

#define REG_ALIGN (4096)
ncclResult_t nccl_ucx_rma_regmr(void* comm, void* data, int size, int type,
                                void** mhandle)
{
  nccl_ucx_rma_ctx_t   *ctx = (nccl_ucx_rma_ctx_t*)comm;
  uint64_t             addr = (uint64_t)data;
  ucp_mem_map_params_t mmap_params;
  ucx_rma_mhandle_t    *mh;
  uint64_t             reg_addr, reg_size;
  void                 *rkey_buf;
  int                  i;
  
  for (i = 0; i < NCCL_UCX_RMA_MAX_MHANDLES; i++) {
    if (ctx->mh[i] == NULL) {
      break;
    }
  }
  if (i == NCCL_UCX_RMA_MAX_MHANDLES) {
    WARN("NET UCX/RMA: too many mhandles");
    return ncclSystemError;
  }

  NCCLCHECK(ncclIbMalloc((void**)&mh, sizeof(ucx_rma_mhandle_t)));
  reg_addr = addr & (~(REG_ALIGN - 1));
  reg_size = addr + size - reg_addr;
  reg_size = ROUNDUP(reg_size, REG_ALIGN);

  mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH; 
  mmap_params.address    = (void*)reg_addr;
  mmap_params.length     = reg_size;  
  mh->mem_type = type;
#if UCP_API_VERSION >= UCP_VERSION(1, 10)
  mh->mem_type = (type == NCCL_PTR_HOST)? UCS_MEMORY_TYPE_HOST: UCS_MEMORY_TYPE_CUDA;
  mmap_params.field_mask  |= UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE; 
  mmap_params.memory_type = mh->mem_type;
#endif

  UCXCHECK(ucp_mem_map(ctx->ctx, &mmap_params, &mh->ucp_memh));
  UCXCHECK(ucp_rkey_pack(ctx->ctx, mh->ucp_memh, &rkey_buf, &mh->rkey_buf.rkey_buf_size));
  if (mh->rkey_buf.rkey_buf_size > MAX_UCX_RKEY_BUF_SIZE) {
    WARN("NET UCX/RMA: rkey_buf is too large");
    ucp_mem_unmap(ctx->ctx, mh->ucp_memh);
    ucp_rkey_buffer_release(rkey_buf);
    free(mh);
    return ncclSystemError;
  }
  memcpy(mh->rkey_buf.buf, rkey_buf, mh->rkey_buf.rkey_buf_size);

  if (ctx->gpuFlush.enabled) {
    UCXCHECK(ucp_ep_rkey_unpack(ctx->gpuFlush.flush_ep, rkey_buf, &mh->rkey));
  }
  
  mh->rkey_buf.index = i;
  mh->rkey_buf.send  = 0;
  mh->rkey_buf.id    = ctx->num_mh;
  ctx->mh[i]   = mh;
  ctx->num_mh += 1;
  *mhandle = mh;
  ucp_rkey_buffer_release(rkey_buf);

  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_regmr_dmabuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
	return nccl_ucx_rma_regmr(comm, data, size, type, mhandle);
}

ncclResult_t nccl_ucx_rma_deregmr(void* comm, void* mhandle)
{
  nccl_ucx_rma_ctx_t *ctx = (nccl_ucx_rma_ctx_t*)comm;
  ucx_rma_mhandle_t  *mh  = (ucx_rma_mhandle_t*)mhandle;

  ctx->mh[mh->rkey_buf.index] = NULL;
  if (ctx->gpuFlush.enabled) {
      ucp_rkey_destroy(mh->rkey);
  }
  ucp_mem_unmap(ctx->ctx, mh->ucp_memh);
  free(mh);

  return ncclSuccess;
}

ncclResult_t ucx_rma_get_request(nccl_ucx_rma_request_t* reqs, int* req_id)
{
  nccl_ucx_rma_request_t *r;
  int i;

  for (i = 0; i < MAX_REQUESTS; i++) {
    r = reqs + i;
    if (r->used == 0) {
      r->used = 1;
      r->type = 0;
      r->done = NCCL_UCX_RMA_REQUEST_INPROGRESS;
      r->size = -1;
      r->free = 0;
      r->st   = NULL;
      *req_id = i;
      return ncclSuccess;
    }
  }
  WARN("NET/UCX_RMA: unable to allocate requests");
  *req_id = -1;

  return ncclInternalError;
}

static void nccl_ucx_rma_ep_flush_cb(void *request, ucs_status_t status)
{
  return;
}

static void nccl_ucx_rma_gdr_flush_cb(void *request, ucs_status_t status)
{
  nccl_ucx_flush_request_t *req = (nccl_ucx_flush_request_t*)request;

  req->req->done = NCCL_UCX_RMA_REQUEST_DONE;
  return;
}

/*
 * nccl_ucx_rma_send_check prepeares send communictor to be used for actual data
 * communication and consists of multiple stages:
 */
enum {
  NCCL_UCX_RMA_SCOMM_NOT_READY = 0, /* initial comm state, only ucp worker is present
                                     * wait for remote worker addr and create ep
                                     * notify peer that endpoint has been created
                                     */
  NCCL_UCX_RMA_SCOMM_EP_CREATED,    /* endpoint is created but it's not gurantee that
                                     * wireup is done. ucp_ep_flush is used to finish
                                     * wireup process
                                     */
  NCCL_UCX_RMA_SCOMM_EP_FLUSH_WAIT, /* ep flush is in progress */
  NCCL_UCX_RMA_SCOMM_READY          /* communicator is ready, notify peer */
};

static ncclResult_t nccl_ucx_rma_send_check(nccl_ucx_rma_send_comm_t *comm)
{
  ucs_status_t st;

  ucp_worker_progress(comm->super.worker);
  if (comm->super.ready == NCCL_UCX_RMA_SCOMM_NOT_READY) {
    NCCLCHECK(nccl_ucx_rma_init_ep(&comm->super.sock, comm->super.worker, &comm->ep, 0));
    if (comm->ep == NULL) {
      return ncclSuccess;
    }
    NCCLCHECK(ncclSocketRecv(&comm->super.sock, &comm->rem_am_id, sizeof(int)));
    comm->super.ready = NCCL_UCX_RMA_SCOMM_EP_CREATED;
  }

  if (comm->super.ready == NCCL_UCX_RMA_SCOMM_EP_CREATED) {
    comm->super.check_req = ucp_ep_flush_nb(comm->ep, 0, nccl_ucx_rma_ep_flush_cb);

  if (comm->super.check_req == NULL) {
      comm->super.ready = NCCL_UCX_RMA_SCOMM_READY;
      NCCLCHECK(ncclSocketSend(&comm->super.sock, &comm->super.ready, sizeof(int)));
    } else if (UCS_PTR_IS_ERR(comm->super.check_req)) {
      return ncclSystemError;
    } else {
      comm->super.ready = NCCL_UCX_RMA_SCOMM_EP_FLUSH_WAIT;
    }
  }

  if (comm->super.ready == NCCL_UCX_RMA_SCOMM_EP_FLUSH_WAIT) {
    st = ucp_request_check_status(comm->super.check_req);
    if (st != UCS_INPROGRESS) {
      ucp_request_free(comm->super.check_req);
      comm->super.ready = NCCL_UCX_RMA_SCOMM_READY; 
      NCCLCHECK(ncclSocketSend(&comm->super.sock, &comm->super.ready, sizeof(int)));
    }
  }

  return ncclSuccess;
}

/*
 * nccl_ucx_rma_recv_check prepeares recv communictor to be used for actual data
 * communication and consists of multiple stages:
 */
enum {
  NCCL_UCX_RMA_RCOMM_SEND_CONN_INFO = 0, /* initial stage, send worker address to peer */ 
  NCCL_UCX_RMA_RCOMM_WAIT_SCOMM,         /* wait for send communicator ready notification */
  NCCL_UCX_RMA_RCOMM_READY,              /* recv comm ready */
};

static ncclResult_t nccl_ucx_rma_recv_check(nccl_ucx_rma_recv_comm_t *comm)
{
  int bytes = 0;
  int rem_comm_state;

  ucp_worker_progress(comm->super.worker);

  if (comm->super.ready == NCCL_UCX_RMA_RCOMM_SEND_CONN_INFO) {
    NCCLCHECK(nccl_ucx_rma_send_worker_address(comm->super.worker, &comm->super.sock));
    NCCLCHECK(ncclSocketSend(&comm->super.sock, &comm->super.id, sizeof(int)));
    comm->super.ready = NCCL_UCX_RMA_RCOMM_WAIT_SCOMM;
  }

  if (comm->super.ready == NCCL_UCX_RMA_RCOMM_WAIT_SCOMM) {
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->super.sock, &rem_comm_state,
                            sizeof(int), &bytes));
    if (bytes == 0) {
      return ncclSuccess;
    }

    NCCLCHECK(ncclSocketWait(NCCL_SOCKET_RECV, &comm->super.sock, &rem_comm_state,
                        sizeof(int), &bytes));
    if (rem_comm_state == NCCL_UCX_RMA_SCOMM_READY) {
      comm->super.ready = NCCL_UCX_RMA_RCOMM_READY;
    } else {
      WARN("Unexpected socket msg %d (%d)", rem_comm_state, NCCL_UCX_RMA_SCOMM_READY);
      return ncclSystemError;
    }    
  }

  return ncclSuccess;
}

static void nccl_ucx_rma_am_isend_cb(void *request, ucs_status_t status)
{
  nccl_ucx_am_request_t *req = (nccl_ucx_am_request_t*)request;

  req->req->done |= NCCL_UCX_RMA_REQUEST_AM_DONE;
  return;
}

static void nccl_ucx_rma_put_isend_cb(void *request, ucs_status_t status, void *data)
{
  nccl_ucx_rma_request_t *req = (nccl_ucx_rma_request_t*)data;

  req->done |= NCCL_UCX_RMA_REQUEST_PUT_DONE;
  return;
}

ncclResult_t nccl_ucx_rma_isend(void *send_comm, void *data, int size, int tag,
                                void *mhandle, void **request)
{
  nccl_ucx_rma_send_comm_t     *comm = (nccl_ucx_rma_send_comm_t*)send_comm;
  ucx_rma_mhandle_t            *mh   = (ucx_rma_mhandle_t*)mhandle;
  volatile ucx_rma_send_fifo_t *slot;
  volatile uint32_t            *ready_ptr;
  volatile int                 *rkey_id;
  volatile int                 *rkey_index;
  nccl_ucx_rma_request_t       *req;
  ucs_status_ptr_t             st;
  int                          req_id;
  ucp_request_param_t          req_param;

  if (comm->super.ready != NCCL_UCX_RMA_SCOMM_READY) {
    NCCLCHECK(nccl_ucx_rma_send_check(comm));
    if (comm->super.ready != NCCL_UCX_RMA_SCOMM_READY) {
      *request = NULL;
      return ncclSuccess;
    }
  }

  slot       = comm->fifo + (comm->fifo_head % MAX_REQUESTS);
  ready_ptr  = &slot->ready;
  rkey_id    = &slot->rkey_id;
  rkey_index = &slot->rkey_idx;

  if ((*ready_ptr == 0) ||
      (comm->rkeys[*rkey_index].id != *rkey_id)) {
    ucp_worker_progress(comm->super.worker);
    *request = NULL;
    return ncclSuccess;
  }

  NCCLCHECK(ucx_rma_get_request(comm->super.reqs, &req_id));
  req = &(comm->super.reqs[req_id]);
  req->size = size;

  req_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                           UCP_OP_ATTR_FIELD_REQUEST  |
                           UCP_OP_ATTR_FIELD_USER_DATA;
  req_param.cb.send      = nccl_ucx_rma_put_isend_cb;
  req_param.user_data    = req;
  req_param.request      = &req->used;
#if UCP_API_VERSION >= UCP_VERSION(1,10)
  if (mh) {
    req_param.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    req_param.memory_type  =  mh->mem_type;
  }
#endif
  
  st  = ucp_put_nbx(comm->ep, data, size, slot->addr,
                    comm->rkeys[*rkey_index].rkey,
                    &req_param);

  if (UCS_PTR_IS_ERR(st)) {
    WARN("NET/UCX_RMA: isend pub_nb failed");
    return ncclInternalError;
  } else if (st  == NULL) {
    req->done |= NCCL_UCX_RMA_REQUEST_PUT_DONE;
  }

  ucp_worker_fence(comm->super.worker);
  req->am_msg = (((uint64_t)slot->req_id) << 32) | ((uint64_t)size);
  req->st = ucp_am_send_nb(comm->ep, comm->rem_am_id, &req->am_msg, 8,
                           ucp_dt_make_contig(1), nccl_ucx_rma_am_isend_cb, 0);

  if (req->st == NULL) {
    req->done |= NCCL_UCX_RMA_REQUEST_AM_DONE;
  } else if (UCS_PTR_IS_PTR(req->st)) {
    nccl_ucx_am_request_t *am_req = (nccl_ucx_am_request_t*)req->st;
    am_req->req = req;
  } else {
    WARN("NET/UCX_RMA: isend am_send_nb failed");
  }

  req->seq = slot->seq;
  slot->ready = 0;
  slot->addr  = 0ULL;
  slot->size  = 0;
  slot->seq   = 0;
  comm->fifo_head++;

  req->worker = comm->super.worker;
  req->type   = UCX_RMA_REQ_TYPE_SEND;
  *request = req;
  return ncclSuccess;
}

static void nccl_ucx_rma_dummy_am_cb(void *request, ucs_status_t status)
{
  return;
}

ncclResult_t nccl_ucx_rma_post_fifo(nccl_ucx_rma_recv_comm_t *comm,
                                    ucx_rma_mhandle_t *mh,
                                    uint64_t addr, int size, int req_id)
{
  ucx_rma_send_fifo_t    *local_elem;
  nccl_ucx_rma_request_t *req;
  uint64_t               remote_addr;
  ucs_status_t           st;

  if (!mh->rkey_buf.send) {
    req = &(comm->super.reqs[req_id]);
    req->st = ucp_am_send_nb(comm->ep, comm->rem_am_id, &mh->rkey_buf,
                             sizeof(nccl_ucx_rma_rkey_buf_t), ucp_dt_make_contig(1),
                             nccl_ucx_rma_dummy_am_cb, 0);
    if (UCS_PTR_IS_ERR(req->st)) {
      WARN("NET/UCX_RMA: am_send_nb failed");
      return ncclInternalError;
    }
    mh->rkey_buf.send = 1;
  }

  local_elem = comm->rem_fifo.elems + (comm->rem_fifo.tail % MAX_REQUESTS);
  local_elem->addr     = addr;
  local_elem->ready    = 1;
  local_elem->size     = size;
  local_elem->seq      = comm->rem_fifo.tail;
  local_elem->rkey_idx = mh->rkey_buf.index;
  local_elem->rkey_id  = mh->rkey_buf.id;
  local_elem->req_id   = req_id;

  remote_addr = comm->rem_fifo.addr + (comm->rem_fifo.tail % MAX_REQUESTS) *
                                      sizeof(ucx_rma_send_fifo_t);
  st = ucp_put_nbi(comm->ep, (void*)local_elem, sizeof(ucx_rma_send_fifo_t),
                   remote_addr, comm->rem_fifo.rkey);
  if (st < 0) {
    WARN("ucx_rma post_fifo pub_nbi failed %d", (int)st);
    return ncclInternalError;
  }

  comm->rem_fifo.tail++;

  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_irecv(void *recv_comm, int n, void **data,int *tags, int *sizes,
                                void **mhandle, void **request)
{
  nccl_ucx_rma_recv_comm_t *comm = (nccl_ucx_rma_recv_comm_t*)recv_comm;
  ucx_rma_mhandle_t        *mh   = (ucx_rma_mhandle_t*)mhandle[0];
  nccl_ucx_rma_request_t   *req;
  int                      req_id;

  if (comm->super.ready != NCCL_UCX_RMA_RCOMM_READY) {
    NCCLCHECK(nccl_ucx_rma_recv_check(comm));
  }

  if (comm->super.ready != NCCL_UCX_RMA_RCOMM_READY) {
    *request = NULL;
    return ncclSuccess;
  }
  
  NCCLCHECK(ucx_rma_get_request(comm->super.reqs, &req_id));
  req = &comm->super.reqs[req_id];

  req->seq = comm->rem_fifo.tail;
  NCCLCHECK(nccl_ucx_rma_post_fifo(comm, mh, (uint64_t)data[0], sizes[0],  req_id));
  ucp_worker_progress(comm->super.worker);
  req->worker = comm->super.worker;
  req->type   = UCX_RMA_REQ_TYPE_RECV;
  *request = req;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_iflush(void* recv_comm, int n, void** data, int* sizes,
                                void** mhandle, void ** request)
{
  nccl_ucx_rma_recv_comm_t *comm = (nccl_ucx_rma_recv_comm_t*)recv_comm;
  ucx_rma_mhandle_t        *mh   = (ucx_rma_mhandle_t*)mhandle[0];
  nccl_ucx_rma_request_t   *req;
  int                      req_id;

  *request = NULL;
  int last = -1;
  for (int i=0; i<n; i++) if (sizes[i]) last = i;
  if (comm->super.gpuFlush.enabled == 0 || last == -1) return ncclSuccess;

  NCCLCHECK(ucx_rma_get_request(comm->super.reqs, &req_id));
  req = &comm->super.reqs[req_id];

  req->st = ucp_get_nb(comm->super.gpuFlush.flush_ep, &comm->super.gpuFlush.hostMem, 1,
                   (uint64_t)data, mh->rkey, nccl_ucx_rma_gdr_flush_cb);
  if (UCS_PTR_IS_ERR(req->st)) {
    WARN("ucx_flush: unable to read data (%s)", ucs_status_string(UCS_PTR_STATUS(req)));
    return ncclSystemError;
  } else if (req->st == NULL) {
    return ncclSuccess;
  }
  nccl_ucx_flush_request_t *flush_req = (nccl_ucx_flush_request_t*)req->st;
  flush_req->req = req;

  req->worker = comm->super.worker;
  req->type   = UCX_RMA_REQ_TYPE_FLUSH;
  *request    = req;

  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_test(void *request, int *done, int *size)
{
  nccl_ucx_rma_request_t *req = (nccl_ucx_rma_request_t*)request;
  unsigned p;

  *done = 0;
  do {
    if (req->done == NCCL_UCX_RMA_REQUEST_DONE) {
      *done = 1;
      if (size) {
        *size = req->size;
      }
      if (req->st != NULL) {
        ucp_request_free(req->st);
      }
      req->used = 0;
      return ncclSuccess;
    }

    p = ucp_worker_progress(req->worker);
  } while (p);

  return ncclSuccess;
}

static void wait_close(ucp_worker_h worker, nccl_ucx_rma_request_t *req)
{
  ucs_status_t status;

  if (UCS_PTR_IS_PTR(req)) {
    do {
      ucp_worker_progress(worker);
      status = ucp_request_check_status(req);
    } while(status == UCS_INPROGRESS);
    ucp_request_free(req);
  } else if (req != NULL) {
      WARN("Failed to close UCX endpoint");
  }
}

ncclResult_t nccl_ucx_rma_close_send(void *send_comm)
{
  nccl_ucx_rma_send_comm_t *comm = (nccl_ucx_rma_send_comm_t*) send_comm;
  void *close_req;
  int close = 1;
  int  i;

  if (send_comm) {
    ucp_mem_unmap(comm->super.ctx, comm->fifo_memh);

    for (i = 0; i < comm->super.num_mh; i++) {
      if (comm->rkeys[i].rkey) {
        ucp_rkey_destroy(comm->rkeys[i].rkey);
      }
    }
    if (comm->ep) {
      close_req = ucp_ep_close_nb(comm->ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->super.worker, close_req);
    }
    NCCLCHECK(ncclSocketSend(&comm->super.sock, &close, sizeof(int)));
    nccl_ucx_free_worker(comm->super.worker);
    free(comm);
  }

  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_close_recv(void *recv_comm)
{
  nccl_ucx_rma_recv_comm_t *comm = (nccl_ucx_rma_recv_comm_t*)recv_comm;
  void *close_req;
  int debug = 1;
  int close = 1;

  if (recv_comm) {
    ucp_rkey_destroy(comm->rem_fifo.rkey);
    if (comm->super.gpuFlush.enabled) {
      close_req = ucp_ep_close_nb(comm->super.gpuFlush.flush_ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->super.worker, close_req);
    }
    if (comm->ep) {
      close_req = ucp_ep_close_nb(comm->ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->super.worker, close_req);
    }
    NCCLCHECK(ncclSocketSend(&comm->super.sock, &close, sizeof(int)));
    nccl_ucx_free_worker(comm->super.worker);
    free(comm);
  }
  
  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_close_listen(void *listen_comm)
{
  nccl_ucx_rma_listen_comm_t *comm = (nccl_ucx_rma_listen_comm_t *)listen_comm;

  if (comm) {
    close(comm->sock.fd);
    free(comm);
  }
  
  return ncclSuccess;
}

ncclNet_v7_t ucxRmaPlugin_v7 = {
  .name = "UCX-RMA",
  .init = nccl_ucx_rma_init,
  .devices = nccl_ucx_rma_devices,
  .getProperties = nccl_ucx_rma_get_properties,
  .listen = nccl_ucx_rma_listen,
  .connect = nccl_ucx_rma_connect,
  .accept = nccl_ucx_rma_accept,
  .regMr = nccl_ucx_rma_regmr,
  .regMrDmaBuf = nccl_ucx_rma_regmr_dmabuf,
  .deregMr = nccl_ucx_rma_deregmr,
  .isend = nccl_ucx_rma_isend,
  .irecv = nccl_ucx_rma_irecv,
  .iflush = nccl_ucx_rma_iflush,
  .test = nccl_ucx_rma_test,
  .closeSend = nccl_ucx_rma_close_send,
  .closeRecv = nccl_ucx_rma_close_recv,
  .closeListen = nccl_ucx_rma_close_listen,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */

};

ncclNet_v6_t ucxRmaPlugin_v6 = {
  .name = "UCX-RMA",
  .init = nccl_ucx_rma_init,
  .devices = nccl_ucx_rma_devices,
  .getProperties = nccl_ucx_rma_get_properties_v6,
  .listen = nccl_ucx_rma_listen,
  .connect = nccl_ucx_rma_connect_v6,
  .accept = nccl_ucx_rma_accept_v6,
  .regMr = nccl_ucx_rma_regmr,
  .regMrDmaBuf = nccl_ucx_rma_regmr_dmabuf,
  .deregMr = nccl_ucx_rma_deregmr,
  .isend = nccl_ucx_rma_isend,
  .irecv = nccl_ucx_rma_irecv,
  .iflush = nccl_ucx_rma_iflush,
  .test = nccl_ucx_rma_test,
  .closeSend = nccl_ucx_rma_close_send,
  .closeRecv = nccl_ucx_rma_close_recv,
  .closeListen = nccl_ucx_rma_close_listen
};

ncclNet_v5_t ucxRmaPlugin_v5 = {
  .name = "UCX-RMA",
  .init = nccl_ucx_rma_init,
  .devices = nccl_ucx_rma_devices,
  .getProperties = nccl_ucx_rma_get_properties_v6,
  .listen = nccl_ucx_rma_listen,
  .connect = nccl_ucx_rma_connect_v6,
  .accept = nccl_ucx_rma_accept_v6,
  .regMr = nccl_ucx_rma_regmr,
  .deregMr = nccl_ucx_rma_deregmr,
  .isend = nccl_ucx_rma_isend,
  .irecv = nccl_ucx_rma_irecv,
  .iflush = nccl_ucx_rma_iflush,
  .test = nccl_ucx_rma_test,
  .closeSend = nccl_ucx_rma_close_send,
  .closeRecv = nccl_ucx_rma_close_recv,
  .closeListen = nccl_ucx_rma_close_listen
};
