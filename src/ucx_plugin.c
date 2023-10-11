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
  ucp_worker_h  worker; /* ucx_worker created on ctx, worker can be shared between
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

struct ep_list {
  struct ncclSocket *sock;
  struct ep_list *next;
};

/**
 * Connection descriptor. Used to store all opened connections.
 */
struct nccl_ucx_worker {
  ucp_context_h  ctx;      /* ucp_context bounded to specific device */
  ucp_worker_h   worker;   /* ucp worker associated with ctx */
  int            count;    /* number of connections that uses this worker */
  struct ep_list *eps;     /* oob conection to all endpoints that were opened on this worker */
  ucp_tag_t      last_tag; /* tag that last created connection uses */
};
static struct nccl_ucx_worker workers[MAX_IB_DEVS];

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
  ucp_worker_h    worker;        /* ucp worker associated with ctx */
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

static ncclResult_t GetSocketAddr(union ncclSocketAddress *addr) {
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

#define UCX_SHARED_WORKER
static ncclResult_t ucx_get_ctx_and_worker(int dev, ucp_context_h *ctx,
                                           ucp_worker_h *worker,
                                           ucp_tag_t *newtag) {
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
  int i, dummy, count;
  struct ep_list *ep, *cur;

  pthread_mutex_lock(&nccl_ucx_lock);
  for(i = 0; i < ncclNIbDevs; i++) {
    if (workers[i].count > 0 && worker == workers[i].worker) {
      count = --workers[i].count;
      break;
    }
  }
  pthread_mutex_unlock(&nccl_ucx_lock);

  if (i < ncclNIbDevs && count == 0) {
    ep = workers[i].eps;
    while(ep){
      cur = ep;
      NCCLCHECK(ncclSocketRecv(ep->sock, &dummy, sizeof(int)));
      ep = ep->next;
      close(cur->sock->fd);
      free(cur);
    }
    ucp_worker_destroy(workers[i].worker);
    ucp_cleanup(workers[i].ctx);
    workers[i].eps    = NULL;
    workers[i].worker = NULL;
    workers[i].ctx    = NULL;
  }

  return ncclSuccess;
}

static ncclResult_t nccl_ucx_add_ep(ucp_worker_h worker, struct ncclSocket *sock) {
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

ncclResult_t nccl_ucx_init(ncclDebugLogger_t logFunction) {
  if (ncclParamUCXDisable()) return ncclInternalError;

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
  NCCLCHECK(ucx_get_ctx_and_worker(dev, &comm->ctx, &comm->worker, &comm->tag));

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
  ucx_listen_handle_t  *recv_handle = (ucx_listen_handle_t*)handle;
  struct ncclUCXCommStage* stage = &recv_handle->stage;
  ucx_comm_t           *comm;
  ucp_address_t        *my_addr;
  size_t               local_addr_len;
  enum ncclSocketState conState;
  *send_comm = NULL;
  int ready;

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

  NCCLCHECK(ucx_get_ctx_and_worker(dev, &comm->ctx, &comm->worker, &comm->ctag));
  comm->tag              = recv_handle->tag;
  comm->gpuFlush.enabled = 0;
  NCCLCHECK(ucx_worker_get_netaddress(comm->worker, &my_addr, &local_addr_len));
  NCCLCHECK(nccl_ucx_add_ep(comm->worker, &comm->sock));
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
  socklen_t          socklen = sizeof(struct sockaddr_in);
  struct ncclUCXCommStage* stage = &l_comm->stage;
  ucx_comm_t         *r_comm = (ucx_comm_t *)stage->comm;
  size_t             peer_addr_len;
  ucp_address_t      *peer_addr;
  ucp_ep_params_t    ep_params;
  struct sockaddr_in sockaddr;
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

  r_comm->ctx    = l_comm->ctx;
  r_comm->worker = l_comm->worker;
  r_comm->tag    = l_comm->tag;

  ucx_request_init(r_comm);

  NCCLCHECK(ncclSocketRecv(&r_comm->sock, &peer_addr_len, sizeof(size_t)));
  peer_addr = malloc(peer_addr_len);
  if (peer_addr == NULL) {
    return ncclSystemError;
  }

  NCCLCHECK(ncclSocketRecv(&r_comm->sock, peer_addr, peer_addr_len));
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
  ep_params.address    = peer_addr;
  UCXCHECK(ucp_ep_create(r_comm->worker, &ep_params, &r_comm->ep));
  NCCLCHECK(ncclSocketRecv(&r_comm->sock, &r_comm->ctag, sizeof(ucp_tag_t)));

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
  static const size_t entries = sizeof(comm->reqs) / sizeof(*comm->reqs);
  ucx_request_t *req;

  req = comm->free_req;
  if (req == NULL) {
    WARN("NET/UCX: unable to allocate NCCL request");
    return NULL;
  }

  comm->free_req = req->next;
  req->worker  = comm->worker;
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
  connect_msg_t       *msg;
  ucp_ep_params_t     ep_params;
  void                *ucp_req;
  ucs_status_t        status;

  ucp_worker_progress(comm->worker);

  msg_tag = ucp_tag_probe_nb(comm->worker, comm->ctag, tag_mask, 1, &info_tag);
  if (msg_tag == NULL) {
    return ncclSuccess;
  }

  msg                 = malloc(info_tag.length);
  params.op_attr_mask = 0;
  ucp_req = ucp_tag_msg_recv_nbx(comm->worker, msg, info_tag.length,
                                 msg_tag, &params);
  if (UCS_PTR_IS_ERR(ucp_req)) {
    WARN("Unable to receive connect msg (%s)",
         ucs_status_string(UCS_PTR_STATUS(ucp_req)));
    free(msg);
    return ncclSystemError;
  } else if (ucp_req != NULL) {
    do {
      ucp_worker_progress(comm->worker);
      status = ucp_request_check_status(ucp_req);
    } while (status == UCS_INPROGRESS);
    assert(status == UCS_OK);
  }

  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
  ep_params.address    = (ucp_address_t*)(msg + 1);
  UCXCHECK(ucp_ep_create(comm->worker, &ep_params, &comm->ep));
  comm->ready = 1;
  free(msg);

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

  NCCLCHECK(ucx_worker_get_netaddress(comm->worker, &my_addr, &local_addr_len));
  nccl_ucx_add_ep(comm->worker, &comm->sock);

  msg_len             = sizeof(connect_msg_t) + local_addr_len;
  comm->msg           = calloc(1, msg_len);
  comm->msg->addr_len = local_addr_len;
  memcpy(comm->msg + 1, my_addr, local_addr_len);

  params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                        UCP_OP_ATTR_FIELD_USER_DATA;
  params.cb.send      = check_handler;
  params.user_data    = comm;

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
  ucp_worker_progress(comm->worker);
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
    params.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    params.memory_type   = mh->mem_type;
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
      params.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMORY_TYPE;
      params.memory_type   = mh[i]->mem_type;
    } else {
      params.op_attr_mask &= ~UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    }

    ucp_req = ucp_tag_recv_nbx(comm->worker, data[i], sizes[i],
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
      wait_close(comm->worker, close_req);
      int close = 1;
      NCCLCHECK(ncclSocketSend(&comm->sock, &close, sizeof(int)));
    }
    nccl_ucx_free_worker(comm->worker);
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
      wait_close(comm->worker, close_req);
    }
    if (comm->ep) {
      close_req = ucp_ep_close_nb(comm->ep, UCP_EP_CLOSE_MODE_FLUSH);
      wait_close(comm->worker, close_req);
      int close=1;
      NCCLCHECK(ncclSocketSend(&comm->sock, &close, sizeof(int)));
    }
    nccl_ucx_free_worker(comm->worker);
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
