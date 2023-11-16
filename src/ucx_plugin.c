/*************************************************************************
 * Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

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

NCCL_PARAM(UCXDisable, "UCX_DISABLE", 0);
/* Exclude cuda-related UCX transports */
NCCL_PARAM(UCXCudaDisable, "UCX_CUDA_DISABLE", 1);

extern ncclDebugLogger_t pluginLogFunction;
static const ucp_tag_t tag      = 0x8a000000;
static const ucp_tag_t tag_mask = (uint64_t)(-1);

static int ncclNIbDevs = -1;

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

typedef struct ucx_mhandle {
  ucp_mem_h  ucp_memh;
  ucp_rkey_h rkey;
  int        mem_type;
} ucx_mhandle_t;

ncclResult_t nccl_ucx_devices(int* ndev) {
  *ndev = ncclNIbDevs;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_get_properties(int dev, ncclNetProperties_t* props)
{
  return nccl_p2p_ib_get_properties(ncclIbDevs, dev, props);
}

ncclResult_t nccl_ucx_get_properties_v6(int dev, ncclNetProperties_v6_t* props_v6)
{
  ncclNetProperties_t props;
  ncclResult_t ret = nccl_ucx_get_properties(dev, &props);
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

pthread_mutex_t nccl_ucx_lock = PTHREAD_MUTEX_INITIALIZER;

struct ep_list {
  struct ncclSocket *sock;
  struct ep_list *next;
};

/**
 * Connection descriptor. Used to store all opened connections.
 */
typedef struct nccl_ucx_worker {
  ucp_worker_h   worker;   /* ucp worker associated with ctx */
  ucp_context_h  ctx;      /* ucp_context bounded to specific device */
  struct ep_list *eps;     /* oob conection to all endpoints that were opened on this worker */

  int            count;    /* number of connections that uses this worker */
  int            dev;      /* Managed device */
  pthread_t      thread;   /* Owner thread */

  struct nccl_ucx_worker *next;
} nccl_ucx_worker_t;

/**
 * Listen handle that is sent from receiver to sender through OOB connection
 */
typedef struct ucx_listen_handle {
  union ncclSocketAddress connectAddr; /* reciever socket address */
  uint64_t magic;                      /* random number to help debugging */
  ucp_tag_t               tag;         /* tag that is used to distiguish data that was sent to
                                          this reciever. Required when shared worker is used. */
  struct ncclUCXCommStage stage;
} ucx_listen_handle_t;

/**
 * Listen commincator for UCX plugin.
 */
typedef struct ucx_listen_comm {
  int           dev;    /* device number in ncclIbDevs which will
                         * be used to recieve data */
  struct ncclSocket sock;/* socket for OOB connection */
  ucp_context_h ctx;    /* ucp_context associated with specific device dev */
  nccl_ucx_worker_t *ucx_worker; /* ucx_worker created on ctx, worker can be shared between
                           multiple connections */
  ucp_tag_t     tag;    /* tag that is used to distiguish data that was sent to 
                           this reciever. Required when shared worker is used.*/
  struct ncclUCXCommStage stage;
} ucx_listen_comm_t;

typedef struct connect_msg {
  size_t addr_len;
} connect_msg_t;

struct ucx_comm;

/**
 * Batch of UCX Requests from NCCL perspective
 */
typedef struct ucx_request {
  struct ucx_request *next;    /* Next request in the free list */
  struct ucx_comm    *comm;    /* Owning communicator */
  ucp_worker_h        worker;  /* Worker for all requests */
  int                 pending; /* How many requests are still pending */
  int                 count;   /* How many requests are contained */
  int                 size[NCCL_NET_IB_MAX_RECVS];
} ucx_request_t;

static ucp_tag_t              worker_tags[MAX_IB_DEVS];
static struct nccl_ucx_worker *workers[MAX_IB_DEVS];
static int worker_count = 0;

typedef struct ucx_gpu_flush {
  int      enabled;
  int      hostMem;
  ucp_ep_h flush_ep;
} ucx_gpu_flush_t;

/**
 * Common data member for ucx_comm for send and receive
 * Used to map/unmap memory in nccl_ucx_regmr/nccl_ucx_deregmr
 */
typedef struct ucx_ctx {
  ucp_context_h   ucp_ctx;
  ucx_gpu_flush_t gpuFlush;
} ucx_ctx_t;

/**
 * Sender and Receiver communicator
 */
typedef struct ucx_comm {
  ucp_context_h   ctx;           /* ucp_context bounded to specific device */
  ucx_gpu_flush_t gpuFlush;      /* flushing handle */
  nccl_ucx_worker_t *ucx_worker; /* ucp worker associated with ctx */
  ucp_ep_h        ep;            /* ucp endpoint created on worker */
  ucp_tag_t       tag;           /* datapath tag to filter out message that are not
                                    belong to this connnection */
  ucp_tag_t       ctag;          /* controlpath tag to filter out message that are not
                                    belong to this connnection */
  struct ncclSocket sock;        /* socket for OOB connection */
  int             ready;         /* indicates that receive communicator is fully initialized */
  ucx_request_t   reqs[MAX_REQUESTS]; /* max inflight requests */
  ucx_request_t   *free_req;     /* first request available */
  connect_msg_t   *msg;          /* message to establish reverse connection */
  void            *connect_req;  /* msg request */
} ucx_comm_t;

static void send_handler_nbx(void *request, ucs_status_t status,
                             void *user_data) {
  int *pending = user_data;

  assert(status == UCS_OK);
  assert(*pending > 0);
  (*pending)--;
  ucp_request_free(request);
}

static void recv_handler_nbx(void *request, ucs_status_t status,
                             const ucp_tag_recv_info_t *tag_info,
                             void *user_data) {
  send_handler_nbx(request, status, user_data);
}

static union ncclSocketAddress nccl_ucx_if_addr;
static char if_name[MAX_IF_NAME_SIZE];

static ncclResult_t ucx_config_no_cuda(ucp_config_t *config) {
  char tmp[PATH_MAX];
  const char *ucx_tls;
  ssize_t n;

  ucx_tls = getenv("NCCL_UCX_TLS");
  if (ucx_tls == NULL) {
    ucx_tls = getenv("UCX_TLS");
  }

  if (ucx_tls == NULL) {
    ucx_tls = "^cuda";
  } else if (ucx_tls[0] == '^') {
    /* Negative expression, make sure to keep cuda excluded */
    n = snprintf(tmp, sizeof(tmp), "^cuda,%s", &ucx_tls[1]);
    if (n >= sizeof(tmp)) {
      return ncclInternalError;
    }

    ucx_tls = tmp;
  } else {
    /* Positive expression cannot allow cuda-like transports */
    if ((strstr(ucx_tls, "cuda") != NULL) || (strstr(ucx_tls, "gdr") != NULL)) {
      WARN("Cannot use cuda/gdr transports as part of specified UCX_TLS");
      return ncclInternalError;
    }
  }

  UCXCHECK(ucp_config_modify(config, "TLS", ucx_tls));
  UCXCHECK(ucp_config_modify(config, "RNDV_THRESH", "0"));
  UCXCHECK(ucp_config_modify(config, "RNDV_SCHEME", "get_zcopy"));
  UCXCHECK(
      ucp_config_modify(config, "MEMTYPE_REG_WHOLE_ALLOC_TYPES", "unknown"));
  return ncclSuccess;
}

static ncclResult_t ucx_init_context(ucp_context_h *ctx, int dev) {
  ucp_params_t ucp_params;
  ucp_config_t *config;
  char         ucx_dev_name[PATH_MAX];
  ncclResult_t result;

  snprintf(ucx_dev_name, PATH_MAX, "%s:%d", ncclIbDevs[dev].devName, ncclIbDevs[dev].port);
  UCXCHECK(ucp_config_read("NCCL", NULL, &config));
  UCXCHECK(ucp_config_modify(config, "NET_DEVICES", ucx_dev_name));

  if (ncclParamUCXCudaDisable()) {
    result = ucx_config_no_cuda(config);
    if (result != ncclSuccess) {
      return result;
    }
  }

  memset(&ucp_params, 0, sizeof(ucp_params));
  ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
  ucp_params.features   = UCP_FEATURE_TAG | UCP_FEATURE_RMA;

  UCXCHECK(ucp_init(&ucp_params, config, ctx));
  ucp_config_release(config);

  return ncclSuccess;
}

static ncclResult_t ucx_init_worker(ucp_context_h ctx, ucp_worker_h *worker) {
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

static ncclResult_t ucx_worker_get_netaddress(ucp_worker_h worker,
                                              ucp_address_t **address,
                                              size_t *address_length) {
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

static ncclResult_t ucx_get_ctx_and_worker(int dev, ucp_context_h *ctx,
                                           nccl_ucx_worker_t **ucx_worker,
                                           ucp_tag_t *newtag) {
  pthread_mutex_lock(&nccl_ucx_lock);
  ncclResult_t result;

  if (ncclNIbDevs <= dev) {
    WARN("Device index is too large");
    goto err;
  }

  nccl_ucx_worker_t *w;
  for (w = workers[dev]; w != NULL; w = w->next) {
    assert(w->dev == dev);
    if (w->thread == pthread_self()) {
      break;
    }
  }

  if (w == NULL) {
    w = calloc(1, sizeof(*w));
    if (w == NULL) {
      WARN("Worker allocation failure");
      goto err;
    }

    w->dev    = dev;
    w->thread = pthread_self();
    w->count  = 0;

    result = ucx_init_context(&w->ctx, dev);
    if (result != ncclSuccess) {
        return result;
    }
    ucx_init_worker(w->ctx, &w->worker);
    worker_count++;

    w->next = workers[dev];
    workers[dev] = w;
  }

  *ctx        = w->ctx;
  *ucx_worker = w;
  if (newtag != NULL) {
    *newtag = ++worker_tags[dev];
  }

  ucp_worker_progress(w->worker);
  w->count++;
  pthread_mutex_unlock(&nccl_ucx_lock);
  return ncclSuccess;

err:
  pthread_mutex_unlock(&nccl_ucx_lock);
  return ncclSystemError;
}

static ncclResult_t nccl_ucx_free_worker(nccl_ucx_worker_t *ucx_worker) {
  int dev, dummy, done = 0;
  struct ep_list *ep, *cur;
  struct nccl_ucx_worker *next;
  ncclResult_t result;

  pthread_mutex_lock(&nccl_ucx_lock);
  ucx_worker->count--;
  if (ucx_worker->count == 0) {
      worker_count--;
      done = worker_count == 0;
  }
  pthread_mutex_unlock(&nccl_ucx_lock);

  if (!done) {
    return ncclSuccess;
  }

  for (dev = 0; dev < sizeof(workers) / sizeof(*workers); dev++) {
    for (ucx_worker = workers[dev]; ucx_worker != NULL; ucx_worker = next) {
      next = ucx_worker->next;
      assert(ucx_worker->count == 0);

      ep = ucx_worker->eps;
      while (ep) {
        cur    = ep;
        result = ncclSocketRecv(ep->sock, &dummy, sizeof(int));
        if (result != ncclSuccess) {
          WARN("Failed to receive close for worker cleanup (res:%d)", result);
        }

        ep = ep->next;
        close(cur->sock->fd);
        free(cur);
      }
      ucp_worker_destroy(ucx_worker->worker);
      ucp_cleanup(ucx_worker->ctx);

      free(ucx_worker);
    }

    workers[dev] = NULL;
  }

  return ncclSuccess;
}

static ncclResult_t nccl_ucx_add_ep(nccl_ucx_worker_t *ucx_worker,
                                    struct ncclSocket *sock) {
  struct ep_list *new_ep = (struct ep_list*)malloc(sizeof(struct ep_list));
  if (new_ep == NULL) {
    return ncclSystemError;
  }

  new_ep->sock    = sock;
  new_ep->next    = ucx_worker->eps;
  ucx_worker->eps = new_ep;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_init(ncclDebugLogger_t logFunction) {
  if (ncclParamUCXDisable()) return ncclInternalError;

  for (int i = 0; i < sizeof(worker_tags) / sizeof(*worker_tags); i++) {
    worker_tags[i] = tag;
  }

  return nccl_p2p_ib_init(&ncclNIbDevs, ncclIbDevs, if_name,
                          &nccl_ucx_if_addr, NULL, logFunction);
}

ncclResult_t nccl_ucx_listen(int dev, void *handle, void **listen_comm) {
  ucx_listen_handle_t *my_handle = (ucx_listen_handle_t*)handle;
  ucx_listen_comm_t   *comm      = (ucx_listen_comm_t*)calloc(1, sizeof(*comm));

  NCCL_STATIC_ASSERT(sizeof(ucx_listen_handle_t) < NCCL_NET_HANDLE_MAXSIZE,
                     "UCX listen handle size too large");
  my_handle->magic = NCCL_SOCKET_MAGIC;
  NCCLCHECK(ncclSocketInit(&comm->sock, &nccl_ucx_if_addr, my_handle->magic, ncclSocketTypeNetIb, NULL, 1));
  NCCLCHECK(ncclSocketListen(&comm->sock));
  NCCLCHECK(ncclSocketGetAddr(&comm->sock, &my_handle->connectAddr));
  NCCLCHECK(ucx_get_ctx_and_worker(dev, &comm->ctx, &comm->ucx_worker, &comm->tag));

  comm->dev = dev;
  my_handle->tag = comm->tag;
  *listen_comm = comm;
 
  return ncclSuccess;
}

static void ucx_request_init(ucx_comm_t *comm) {
  static const int entries = sizeof(comm->reqs) / sizeof(*comm->reqs);

  comm->free_req = NULL;
  for (int i = entries - 1; i >= 0; i--) {
      comm->reqs[i].comm = comm;
      comm->reqs[i].next = comm->free_req;
      comm->free_req = &comm->reqs[i];
  }
}

ncclResult_t nccl_ucx_connect(int dev, void *handle, void **send_comm, ncclNetDeviceHandle_t** sendDevComm) {
  ucx_listen_handle_t     *recv_handle = (ucx_listen_handle_t*)handle;
  struct ncclUCXCommStage *stage       = &recv_handle->stage;
  ucx_comm_t              *comm        = stage->comm;
  ucp_address_t           *my_addr;
  size_t                  local_addr_len;
  int                     ready;

  *send_comm = NULL;

  if (stage->state == ncclUCXCommStateConnect) goto ucx_connect_check;

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(ucx_comm_t)));
  NCCLCHECK(ncclSocketInit(&comm->sock, &recv_handle->connectAddr, recv_handle->magic, ncclSocketTypeNetIb, NULL, 1));
  stage->comm = comm;
  stage->state = ncclUCXCommStateConnect;
  NCCLCHECK(ncclSocketConnect(&comm->sock));
  ucx_request_init(comm);

ucx_connect_check:
  /* since ncclSocketConnect is async, we must check if connection is complete */
  NCCLCHECK(ncclSocketReady(&comm->sock, &ready));
  if (!ready) return ncclSuccess;

  NCCLCHECK(ucx_get_ctx_and_worker(dev, &comm->ctx, &comm->ucx_worker, &comm->ctag));
  comm->tag              = recv_handle->tag;
  comm->gpuFlush.enabled = 0;
  NCCLCHECK(ucx_worker_get_netaddress(comm->ucx_worker->worker, &my_addr, &local_addr_len));
  NCCLCHECK(nccl_ucx_add_ep(comm->ucx_worker, &comm->sock));
  INFO(NCCL_NET, "NET/UCX: Worker address length: %zu", local_addr_len);

  NCCLCHECK(ncclSocketSend(&comm->sock, &local_addr_len, sizeof(size_t)));
  NCCLCHECK(ncclSocketSend(&comm->sock, my_addr, local_addr_len));
  NCCLCHECK(ncclSocketSend(&comm->sock, &comm->ctag, sizeof(ucp_tag_t)));

  *send_comm = comm;
  free(my_addr);
  return ncclSuccess;
}

ncclResult_t nccl_ucx_connect_v6(int dev, void *handle, void **send_comm) {
  ncclNetDeviceHandle_v7_t* dev_handle = NULL;
  return nccl_ucx_connect(dev, handle, send_comm, &dev_handle);
}

ncclResult_t nccl_ucx_accept(void *listen_comm, void **recv_comm, ncclNetDeviceHandle_v7_t** recvDevComm)
{
  ucx_listen_comm_t  *l_comm = (ucx_listen_comm_t *)listen_comm;
  struct ncclUCXCommStage* stage = &l_comm->stage;
  ucx_comm_t         *r_comm = (ucx_comm_t *)stage->comm;
  size_t             peer_addr_len;
  ucp_address_t      *peer_addr;
  ucp_ep_params_t    ep_params;
  int                ready;

  *recv_comm = NULL;
  if (stage->state == ncclUCXCommStateAccept) goto ucx_accept_check;

  NCCLCHECK(ncclIbMalloc((void**)&r_comm, sizeof(ucx_comm_t)));
  stage->comm = r_comm;
  stage->state = ncclUCXCommStateAccept;
  l_comm->sock.asyncFlag = 1;
  r_comm->sock.asyncFlag = 1;

  NCCLCHECK(ncclSocketInit(&r_comm->sock, NULL, NCCL_SOCKET_MAGIC, ncclSocketTypeUnknown, NULL, 0));
  NCCLCHECK(ncclSocketAccept(&r_comm->sock, &l_comm->sock));
ucx_accept_check:
  NCCLCHECK(ncclSocketReady(&r_comm->sock, &ready));
  if (!ready) return ncclSuccess;

  r_comm->ctx        = l_comm->ctx;
  r_comm->ucx_worker = l_comm->ucx_worker;
  r_comm->tag        = l_comm->tag;

  ucx_request_init(r_comm);

  NCCLCHECK(ncclSocketRecv(&r_comm->sock, &peer_addr_len, sizeof(size_t)));
  peer_addr = malloc(peer_addr_len);
  if (peer_addr == NULL) {
    return ncclSystemError;
  }

  NCCLCHECK(ncclSocketRecv(&r_comm->sock, peer_addr, peer_addr_len));
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
  ep_params.address    = peer_addr;
  UCXCHECK(ucp_ep_create(r_comm->ucx_worker->worker, &ep_params, &r_comm->ep));
  NCCLCHECK(ncclSocketRecv(&r_comm->sock, &r_comm->ctag, sizeof(ucp_tag_t)));

  r_comm->gpuFlush.enabled = (nccl_p2p_gdr_support(l_comm->dev) == ncclSuccess);  
  if (r_comm->gpuFlush.enabled) {
    ucp_address_t *my_addr;
    size_t        local_addr_len;

    NCCLCHECK(ucx_worker_get_netaddress(r_comm->ucx_worker->worker, &my_addr, &local_addr_len));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address    = my_addr;
    UCXCHECK(ucp_ep_create(r_comm->ucx_worker->worker, &ep_params, &r_comm->gpuFlush.flush_ep));
    free(my_addr);
  }

  free(peer_addr);
  *recv_comm = r_comm;

  return ncclSuccess;
}

ncclResult_t nccl_ucx_accept_v6(void *listen_comm, void **recv_comm) {
  ncclNetDeviceHandle_v7_t* dev_handle = NULL;
  return nccl_ucx_accept(listen_comm, recv_comm, &dev_handle);
}

#define REG_ALIGN (4096)
ncclResult_t nccl_ucx_regmr(void* comm, void* data, int size, int type, void** mhandle) {
  ucx_ctx_t *ctx = (ucx_ctx_t*)comm;
  uint64_t  addr = (uint64_t)  data;
  ucp_mem_map_params_t mmap_params;
  ucx_mhandle_t        *mh;
  uint64_t             reg_addr, reg_size;
  size_t               rkey_buf_size;
  void                 *rkey_buf;
  
  NCCLCHECK(ncclIbMalloc((void**)&mh, sizeof(ucx_mhandle_t)));
  reg_addr = addr & (~(REG_ALIGN - 1));
  reg_size = addr + size - reg_addr;
  reg_size = ROUNDUP(reg_size, REG_ALIGN);

  mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH; 
  mmap_params.address    = (void*)reg_addr;
  mmap_params.length     = reg_size;  
  mh->mem_type = (type == NCCL_PTR_HOST)? UCS_MEMORY_TYPE_HOST: UCS_MEMORY_TYPE_CUDA;
  mmap_params.field_mask  |= UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE; 
  mmap_params.memory_type = mh->mem_type;

  UCXCHECK(ucp_mem_map(ctx->ucp_ctx, &mmap_params, &mh->ucp_memh));
  if (ctx->gpuFlush.enabled) {
    UCXCHECK(ucp_rkey_pack(ctx->ucp_ctx, mh->ucp_memh, &rkey_buf, &rkey_buf_size));
    UCXCHECK(ucp_ep_rkey_unpack(ctx->gpuFlush.flush_ep, rkey_buf, &mh->rkey));
    ucp_rkey_buffer_release(rkey_buf);
  }
  
  *mhandle = mh;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_deregmr(void* comm, void* mhandle) {
  ucx_ctx_t     *ctx = (ucx_ctx_t*)comm;
  ucx_mhandle_t *mh  = (ucx_mhandle_t*)mhandle;

  if (ctx->gpuFlush.enabled) {
      ucp_rkey_destroy(mh->rkey);
  }

  ucp_mem_unmap(ctx->ucp_ctx, mh->ucp_memh);
  free(mhandle);

  return ncclSuccess;
}

ncclResult_t nccl_ucx_regmr_dmabuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
	return nccl_ucx_regmr(comm, data, size, type, mhandle);
}

static ucx_request_t *ucx_request_get(ucx_comm_t *comm) {
  ucx_request_t *req = comm->free_req;

  if (req == NULL) {
    WARN("NET/UCX: unable to allocate NCCL request");
    return NULL;
  }

  comm->free_req = req->next;
  req->worker  = comm->ucx_worker->worker;
  req->pending = 0;
  req->count   = 0;
  return req;
}

static void ucx_request_release(ucx_request_t *req) {
    req->next = req->comm->free_req;
    req->comm->free_req = req;
}

static void ucx_request_add(ucx_request_t *req, int size) {
  req->size[req->count] = size;
  req->pending++;
  req->count++;
}

static ncclResult_t ucx_send_check(ucx_comm_t *comm) {
  ucp_request_param_t params;
  ucp_tag_message_h   msg_tag;
  ucp_tag_recv_info_t info_tag;
  ucp_ep_params_t     ep_params;
  void                *ucp_req;
  ucs_status_t        status;

  ucp_worker_progress(comm->ucx_worker->worker);

  if (comm->connect_req != NULL) {
    goto out_check_status;
  }

  msg_tag = ucp_tag_probe_nb(comm->ucx_worker->worker, comm->ctag, tag_mask, 1,
                             &info_tag);
  if (msg_tag == NULL) {
    return ncclSuccess;
  }

  comm->msg = malloc(info_tag.length);
  if (comm->msg == NULL) {
    return ncclSystemError;
  }

  params.op_attr_mask = 0;
  ucp_req = ucp_tag_msg_recv_nbx(comm->ucx_worker->worker, comm->msg,
                                 info_tag.length, msg_tag, &params);
  if (UCS_PTR_IS_ERR(ucp_req)) {
    WARN("Unable to receive connect msg (%s)",
         ucs_status_string(UCS_PTR_STATUS(ucp_req)));
    free(comm->msg);
    comm->msg = NULL;
    return ncclSystemError;
  } else if (ucp_req == NULL) {
    goto out_set_ready;
  }

  assert(comm->connect_req == NULL);
  comm->connect_req = ucp_req;

out_check_status:
  status = ucp_request_check_status(comm->connect_req);
  if (status == UCS_INPROGRESS) {
    return ncclSuccess;
  }

  if (status != UCS_OK) {
    free(comm->msg);
    comm->msg = NULL;
    WARN("Send check requested returned error (%s)", ucs_status_string(status));
    return ncclSystemError;
  }

  ucp_request_free(comm->connect_req);
  comm->connect_req = NULL;

out_set_ready:
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
  ep_params.address    = (ucp_address_t*)(comm->msg + 1);
  UCXCHECK(ucp_ep_create(comm->ucx_worker->worker, &ep_params, &comm->ep));
  comm->ready = 1;
  free(comm->msg);
  comm->msg = NULL;

  return ncclSuccess;
}

static void ucx_recv_set_ready(ucx_comm_t *comm) {
  free(comm->msg);
  comm->msg   = NULL;
  comm->ready = 1;
}

static void check_handler(void *request, ucs_status_t status, void *user_data) {
  assert(status == UCS_OK);
  ucx_recv_set_ready((ucx_comm_t*)user_data);
  ucp_request_free(request);
}

ncclResult_t ucx_recv_check(ucx_comm_t *comm) {
  ucp_request_param_t params;
  ucp_address_t       *my_addr;
  size_t              local_addr_len;
  size_t              msg_len;

  if (comm->connect_req != NULL) {
    goto done;
  }

  NCCLCHECK(ucx_worker_get_netaddress(comm->ucx_worker->worker, &my_addr,
                                      &local_addr_len));
  nccl_ucx_add_ep(comm->ucx_worker, &comm->sock);

  msg_len             = sizeof(connect_msg_t) + local_addr_len;
  comm->msg           = calloc(1, msg_len);
  comm->msg->addr_len = local_addr_len;
  memcpy(comm->msg + 1, my_addr, local_addr_len);

  params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                        UCP_OP_ATTR_FIELD_USER_DATA;
  params.cb.send      = check_handler;
  params.user_data    = comm;

  assert(comm->connect_req == NULL);
  comm->connect_req   = ucp_tag_send_nbx(comm->ep, comm->msg, msg_len,
                                         comm->ctag, &params);
  if (UCS_PTR_IS_ERR(comm->connect_req)) {
    WARN("Unable to send connect message");
    free(comm->msg);
    return ncclSystemError;
  } else if (comm->connect_req == NULL) {
    ucx_recv_set_ready(comm);
    return ncclSuccess;
  }

done:
  ucp_worker_progress(comm->ucx_worker->worker);
  return ncclSuccess;
}

static ucp_tag_t nccl_ucx_ucp_tag(ucp_tag_t comm_tag, uint64_t tag)
{
  assert(tag <= UINT32_MAX);
  assert(comm_tag <= UINT32_MAX);
  return comm_tag + (tag << 32);
}

static ncclResult_t nccl_ucx_isend(void *send_comm, void *data, int size,
                                   int tag, void *mhandle, void **request)
{
  ucx_comm_t         *comm = (ucx_comm_t *)send_comm;
  ucx_mhandle_t      *mh   = (ucx_mhandle_t*)mhandle;
  ucx_request_t      *req;
  void               *ucp_req;
  ucp_request_param_t params;

  if (comm->ready == 0) {
    NCCLCHECK(ucx_send_check(comm));
    if (comm->ready == 0) {
      *request = NULL;
      return ncclSuccess;
    }
  }

  req = ucx_request_get(comm);
  if (req == NULL) {
    return ncclInternalError;
  }

  ucx_request_add(req, size);

  params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                        UCP_OP_ATTR_FIELD_USER_DATA;
  params.cb.send      = send_handler_nbx;
  params.user_data    = &req->pending;
  if (mh) {
    params.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
    params.memh          = mh->ucp_memh;
  }


  ucp_req = ucp_tag_send_nbx(comm->ep, data, size,
                             nccl_ucx_ucp_tag(comm->tag, tag), &params);
  if (UCS_PTR_IS_ERR(ucp_req)) {
    WARN("ucx_isend: unable to send message (%s)",
         ucs_status_string(UCS_PTR_STATUS(ucp_req)));
    return ncclSystemError;
  } else if (ucp_req == NULL) {
    req->pending--;
  }

  *request = req;
  return ncclSuccess;
}

static ncclResult_t nccl_ucx_irecv(void *recv_comm, int n, void **data,
                                   int *sizes, int *tags, void **mhandle,
                                   void **request)
{
  ucx_comm_t         *comm = (ucx_comm_t*)recv_comm;
  ucx_mhandle_t      **mh  = (ucx_mhandle_t**)mhandle;
  void               *ucp_req;
  ucx_request_t      *req;
  ucp_request_param_t params;

  if (comm->ready == 0) {
    NCCLCHECK(ucx_recv_check(comm));
    if (comm->ready == 0) {
      *request = NULL;
      return ncclSuccess;
    }
  }

  if (n > NCCL_NET_IB_MAX_RECVS) {
    WARN("ucx_irecv: posting %d but max is %d", n, NCCL_NET_IB_MAX_RECVS);
    return ncclInternalError;
  }

  req = ucx_request_get(comm);
  if (req == NULL) {
    return ncclInternalError;
  }

  params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                        UCP_OP_ATTR_FIELD_USER_DATA;
  params.cb.recv      = recv_handler_nbx;
  params.user_data    = &req->pending;

  for (int i = 0; i < n; i++) {
    ucx_request_add(req, sizes[i]);

    if (mh[i]) {
      params.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
      params.memh          = mh[i]->ucp_memh;
    } else {
      params.op_attr_mask &= ~UCP_OP_ATTR_FIELD_MEMH;
    }

    ucp_req = ucp_tag_recv_nbx(comm->ucx_worker->worker, data[i], sizes[i],
                               nccl_ucx_ucp_tag(comm->tag, tags[i]), tag_mask,
                               &params);
    if (UCS_PTR_IS_ERR(ucp_req)) {
      WARN("ucx_irecv: unable to post receive %d/%d (%s)", i, n,
           ucs_status_string(UCS_PTR_STATUS(ucp_req)));
      return ncclSystemError;
    } else if (ucp_req == NULL) {
      req->pending--;
    }
  }

  *request = req;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_iflush(void *recv_comm, int n, void **data, int *sizes,
                             void **mhandle, void **request) {
  int last            = -1;
  int size            = 1;
  ucx_comm_t    *comm = (ucx_comm_t *)recv_comm;
  ucx_mhandle_t **mh  = (ucx_mhandle_t**)mhandle;
  ucx_request_t *req;
  void *ucp_req;
  ucp_request_param_t params;

  *request = NULL;
  for (int i=0; i<n; i++) if (sizes[i]) last = i;
  if (comm->gpuFlush.enabled == 0 || last == -1) return ncclSuccess;

  req = ucx_request_get(comm);
  if (req == NULL) {
    return ncclInternalError;
  }

  ucx_request_add(req, size);

  params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                        UCP_OP_ATTR_FIELD_USER_DATA;
  params.cb.send      = send_handler_nbx;
  params.user_data    = &req->pending;
  ucp_req = ucp_get_nbx(comm->gpuFlush.flush_ep, &comm->gpuFlush.hostMem, size,
                        (uint64_t)data[last], mh[last]->rkey, &params);
  if (UCS_PTR_IS_ERR(ucp_req)) {
    WARN("ucx_iflush: unable to read data (%s)",
         ucs_status_string(UCS_PTR_STATUS(ucp_req)));
    return ncclSystemError;
  } else if (ucp_req == NULL) {
    req->pending--;
  }

  *request = req;
  return ncclSuccess;
}

static ncclResult_t nccl_ucx_test(void *request, int *done, int *size) {
  ucx_request_t *req = request;
  unsigned p;

  while (req->pending > 0) {
    p = ucp_worker_progress(req->worker);
    if (!p) {
      *done = 0;
      return ncclSuccess;
    }
  }

  *done = 1;
  if (size != NULL) {
    /* Posted receives have completed */
    memcpy(size, req->size, sizeof(*size) * req->count);
  }

  ucx_request_release(req);
  return ncclSuccess;
}

static void wait_close(ucp_worker_h worker, void *ucp_req) {
  ucs_status_t status;

  if (UCS_PTR_IS_PTR(ucp_req)) {
    do {
      ucp_worker_progress(worker);
      status = ucp_request_check_status(ucp_req);
    } while(status == UCS_INPROGRESS);
    ucp_request_free(ucp_req);
  } else if (ucp_req != NULL) {
    WARN("Failed to close UCX endpoint");
  }
}

ncclResult_t nccl_ucx_close_send(void *send_comm) {
  ucx_comm_t *comm = (ucx_comm_t*)send_comm;
  void *close_req;

  if (comm) {
    if (comm->ep) {
      close_req = ucp_ep_close_nb(comm->ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->ucx_worker->worker, close_req);
      int close = 1;
      NCCLCHECK(ncclSocketSend(&comm->sock, &close, sizeof(int)));
    }
    nccl_ucx_free_worker(comm->ucx_worker);
    free(comm);
  }

  return ncclSuccess;
}

ncclResult_t nccl_ucx_close_recv(void *recv_comm) {
  ucx_comm_t *comm = (ucx_comm_t*)recv_comm;
  void *close_req;

  if (comm) {
    if (comm->gpuFlush.enabled) {
      close_req = ucp_ep_close_nb(comm->gpuFlush.flush_ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->ucx_worker->worker, close_req);
    }
    if (comm->ep) {
      close_req = ucp_ep_close_nb(comm->ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->ucx_worker->worker, close_req);
      int close=1;
      NCCLCHECK(ncclSocketSend(&comm->sock, &close, sizeof(int)));
    }
    nccl_ucx_free_worker(comm->ucx_worker);
    free(comm);
  }

  return ncclSuccess;
}

ncclResult_t nccl_ucx_close_listen(void *listen_comm) {
  ucx_listen_comm_t *comm = (ucx_listen_comm_t *)listen_comm;

  if (comm) {
    close(comm->sock.fd);
    free(comm);
  }
  
  return ncclSuccess;
}

ncclNet_v8_t ucxPlugin_v8 = {
  .name = "UCX",
  .init = nccl_ucx_init,
  .devices = nccl_ucx_devices,
  .getProperties = nccl_ucx_get_properties,
  .listen = nccl_ucx_listen,
  .connect = nccl_ucx_connect,
  .accept = nccl_ucx_accept,
  .regMr = nccl_ucx_regmr,
  .regMrDmaBuf = nccl_ucx_regmr_dmabuf,
  .deregMr = nccl_ucx_deregmr,
  .isend = nccl_ucx_isend,
  .irecv = nccl_ucx_irecv,
  .iflush = nccl_ucx_iflush,
  .test = nccl_ucx_test,
  .closeSend = nccl_ucx_close_send,
  .closeRecv = nccl_ucx_close_recv,
  .closeListen = nccl_ucx_close_listen,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */
};

ncclNet_v7_t ucxPlugin_v7 = {
  .name = "UCX",
  .init = nccl_ucx_init,
  .devices = nccl_ucx_devices,
  .getProperties = nccl_ucx_get_properties,
  .listen = nccl_ucx_listen,
  .connect = nccl_ucx_connect,
  .accept = nccl_ucx_accept,
  .regMr = nccl_ucx_regmr,
  .regMrDmaBuf = nccl_ucx_regmr_dmabuf,
  .deregMr = nccl_ucx_deregmr,
  .isend = nccl_ucx_isend,
  .irecv = nccl_ucx_irecv,
  .iflush = nccl_ucx_iflush,
  .test = nccl_ucx_test,
  .closeSend = nccl_ucx_close_send,
  .closeRecv = nccl_ucx_close_recv,
  .closeListen = nccl_ucx_close_listen,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */
};

ncclNet_v6_t ucxPlugin_v6 = {
  .name = "UCX",
  .init = nccl_ucx_init,
  .devices = nccl_ucx_devices,
  .getProperties = nccl_ucx_get_properties_v6,
  .listen = nccl_ucx_listen,
  .connect = nccl_ucx_connect_v6,
  .accept = nccl_ucx_accept_v6,
  .regMr = nccl_ucx_regmr,
  .regMrDmaBuf = nccl_ucx_regmr_dmabuf,
  .deregMr = nccl_ucx_deregmr,
  .isend = nccl_ucx_isend,
  .irecv = nccl_ucx_irecv,
  .iflush = nccl_ucx_iflush,
  .test = nccl_ucx_test,
  .closeSend = nccl_ucx_close_send,
  .closeRecv = nccl_ucx_close_recv,
  .closeListen = nccl_ucx_close_listen
};

ncclNet_v5_t ucxPlugin_v5 = {
  .name = "UCX",
  .init = nccl_ucx_init,
  .devices = nccl_ucx_devices,
  .getProperties = nccl_ucx_get_properties_v6,
  .listen = nccl_ucx_listen,
  .connect = nccl_ucx_connect_v6,
  .accept = nccl_ucx_accept_v6,
  .regMr = nccl_ucx_regmr,
  .deregMr = nccl_ucx_deregmr,
  .isend = nccl_ucx_isend,
  .irecv = nccl_ucx_irecv,
  .iflush = nccl_ucx_iflush,
  .test = nccl_ucx_test,
  .closeSend = nccl_ucx_close_send,
  .closeRecv = nccl_ucx_close_recv,
  .closeListen = nccl_ucx_close_listen
};
