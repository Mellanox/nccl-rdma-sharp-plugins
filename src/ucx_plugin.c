/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
static const ucp_tag_t tag      = 0xABADBABE;
static const ucp_tag_t tag_mask = (uint64_t)(-1);

static int ncclNIbDevs = -1;

/*
 * If request == REQUEST_COMPLETED_ZERO_LENGTGH:
 *  ucp_send or ucp_recv was completed immediately and worker progress is not needed
 *  message size == 0 and gpu flush is not needed
 * 
 * If request == REQUEST_COMPLETED_NON_ZERO_LENGTH:
 *  ucp_send or ucp_recv was completed immediately and worker progres is not needed
 *  message size > 0 and gpu flush is needed
 * 
 * If request != REQUEST_COMPLETED_ZERO_LENGTGH and request != REQUEST_COMPLETED_NON_ZERO_LENGTH:
 *  normal ucp request.
 */
enum {
  REQUEST_COMPLETED_ZERO_LENGTGH    = 1,
  REQUEST_COMPLETED_NON_ZERO_LENGTH = 2
};

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

typedef struct ucx_request {
  int          completed;
  int          size;
  ucp_worker_h worker;
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
 * Common data member for ucx_send_comm and ucx_recv_comm.
 * Used to map/unmap memory in nccl_ucx_regmr/nccl_ucx_deregmr
 */
typedef struct ucx_ctx {
  ucp_context_h   ucp_ctx;
  ucx_gpu_flush_t gpuFlush;
} ucx_ctx_t;

/**
 * Sender communicator
 */
typedef struct ucx_send_comm {
  ucp_context_h   ctx;        /* ucp_context bounded to specific device */
  ucx_gpu_flush_t gpuFlush;   /* flushing handle */
  ucp_worker_h    worker;     /* ucp worker associated with ctx */
  ucp_ep_h        ep;         /* ucp endpoint created on worker */
  ucp_tag_t       tag;        /* datapath tag to filter out message that are not
                                 belong to this connnection */
  ucp_tag_t       ctag;       /* controlpath tag to filter out message that are not
                                 belong to this connnection */
  struct ncclSocket sock;     /* socket for OOB connection */
  int             ready;      /* indicates that send communicator is fully initialized */
} ucx_send_comm_t;

/**
 * Receiver communicator
 */
typedef struct ucx_recv_comm {
  ucp_context_h   ctx;           /* ucp_context bounded to specific device */
  ucx_gpu_flush_t gpuFlush;      /* flushing handle */
  ucp_worker_h    worker;        /* ucp worker associated with ctx */
  ucp_ep_h        ep;            /* ucp endpoint created on worker */
  ucp_tag_t       tag;           /* datapath tag to filter out message that are not
                                    belong to this connnection */
  ucp_tag_t       ctag;          /* controlpath tag to filter out message that are not
                                    belong to this connnection */
  struct ncclSocket sock;     /* socket for OOB connection */
  int             ready;         /* indicates that receive communicator is fully initialized */
  connect_msg_t   *msg;          /* message to establish reverse connection */
  ucx_request_t   *connect_req;  /* msg request */
} ucx_recv_comm_t;

static void request_init(void *request) {
  ucx_request_t *req = (ucx_request_t*)request;
  req->completed = 0;
}

static void send_handler(void *request, ucs_status_t status) {
  ucx_request_t *req = (ucx_request_t*)request;
  req->completed = 1;
}

static void send_handler_nbx(void *request, ucs_status_t status,
                             void *user_data) {
  ucx_request_t *req = (ucx_request_t*)request;
  req->completed = 1;
}

static void recv_handler(void *request, ucs_status_t status,
                         ucp_tag_recv_info_t *info) {
  ucx_request_t *req = (ucx_request_t*)request;
  req->completed = 1;
}

void recv_handler_nbx(void *request, ucs_status_t status,
                      const ucp_tag_recv_info_t *tag_info,
                      void *user_data) {
  ucx_request_t *req = (ucx_request_t*)request;
  req->completed = 1;
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
  ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES |
                            UCP_PARAM_FIELD_REQUEST_SIZE |
                            UCP_PARAM_FIELD_REQUEST_INIT;
  ucp_params.features     = UCP_FEATURE_TAG |
                            UCP_FEATURE_RMA;
  ucp_params.request_size = sizeof(ucx_request_t);
  ucp_params.request_init = request_init;

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
      break;
    }
  }
  pthread_mutex_unlock(&nccl_ucx_lock);

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

ncclResult_t nccl_ucx_connect(int dev, void *handle, void **send_comm) {
  ucx_listen_handle_t  *recv_handle = (ucx_listen_handle_t*)handle;
  struct ncclUCXCommStage* stage = &recv_handle->stage;
  ucx_send_comm_t      *comm;
  ucp_address_t        *my_addr;
  size_t               local_addr_len;
  enum ncclSocketState conState;
  *send_comm = NULL;
  int ready;

  if (stage->state == ncclUCXCommStateConnect) goto ucx_connect_check;

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(ucx_send_comm_t)));
  NCCLCHECK(ncclSocketInit(&comm->sock, &recv_handle->connectAddr, recv_handle->magic, ncclSocketTypeNetIb, NULL, 1));
  stage->comm = comm;
  stage->state = ncclUCXCommStateConnect;
  NCCLCHECK(ncclSocketConnect(&comm->sock));

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

ncclResult_t nccl_ucx_accept(void *listen_comm, void **recv_comm) {
  ucx_listen_comm_t  *l_comm = (ucx_listen_comm_t *)listen_comm;
  socklen_t          socklen = sizeof(struct sockaddr_in);
  struct ncclUCXCommStage* stage = &l_comm->stage;
  ucx_recv_comm_t    *r_comm = (ucx_recv_comm_t *)stage->comm;
  size_t             peer_addr_len;
  ucp_address_t      *peer_addr;
  ucp_ep_params_t    ep_params;
  struct sockaddr_in sockaddr;
  int                ready;

  *recv_comm = NULL;
  if (stage->state == ncclUCXCommStateAccept) goto ucx_accept_check;

  NCCLCHECK(ncclIbMalloc((void**)&r_comm, sizeof(ucx_recv_comm_t)));
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
  mh->mem_type = type;
#if UCP_API_VERSION >= UCP_VERSION(1, 10)
  mh->mem_type = (type == NCCL_PTR_HOST)? UCS_MEMORY_TYPE_HOST: UCS_MEMORY_TYPE_CUDA;
  mmap_params.field_mask  |= UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE; 
  mmap_params.memory_type = mh->mem_type;
#endif

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

ncclResult_t ucx_send_check(ucx_send_comm_t *comm) {
  ucp_tag_message_h     msg_tag;
  ucp_tag_recv_info_t   info_tag;
  ucx_request_t         *req;
  connect_msg_t         *msg;
  ucp_ep_params_t       ep_params;

  ucp_worker_progress(comm->worker);

  msg_tag = ucp_tag_probe_nb(comm->worker, comm->ctag, tag_mask, 1, &info_tag);
  if (msg_tag == NULL) {
    return ncclSuccess;
  }

  msg = malloc(info_tag.length);
  req = ucp_tag_msg_recv_nb(comm->worker, msg, info_tag.length,
                            ucp_dt_make_contig(1), msg_tag, recv_handler);

  if (UCS_PTR_IS_ERR(req)) {
    WARN("Unable to receive connect msg (%s)", ucs_status_string(UCS_PTR_STATUS(req)));
  } else {
    while (req->completed == 0) {
      ucp_worker_progress(comm->worker);
    }
    req->completed = 0;
    ucp_request_free(req);
  }

  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
  ep_params.address    = (ucp_address_t*)(msg + 1);
  UCXCHECK(ucp_ep_create(comm->worker, &ep_params, &comm->ep));
  comm->ready = 1;
  free(msg);

  return ncclSuccess;
}

ncclResult_t ucx_recv_check(ucx_recv_comm_t *comm) {
  ucp_address_t *my_addr;
  size_t        local_addr_len;
  size_t        msg_len;

  if (comm->connect_req == NULL) {
    NCCLCHECK(ucx_worker_get_netaddress(comm->worker, &my_addr, &local_addr_len));
    nccl_ucx_add_ep(comm->worker, &comm->sock);
    msg_len = sizeof(connect_msg_t) + local_addr_len;
    comm->msg = calloc(1, msg_len);
    comm->msg->addr_len = local_addr_len;
    memcpy(comm->msg + 1, my_addr, local_addr_len);
    comm->connect_req = ucp_tag_send_nb(comm->ep, comm->msg, msg_len,
                                        ucp_dt_make_contig(1), comm->ctag,
                                        send_handler);

    if (UCS_PTR_IS_ERR(comm->connect_req)) {
      WARN("Unable to send connect message");
      return ncclSystemError;
    } else if (comm->connect_req == NULL){
      comm->ready = 1;
      free(comm->msg);
    }

    free(my_addr);
  } else {
    if (comm->connect_req->completed == 1) {
      comm->ready = 1;
      comm->connect_req->completed = 0;
      ucp_request_free(comm->connect_req);
      free(comm->msg);
    }
    else {
      ucp_worker_progress(comm->worker);
    }
  }
  return ncclSuccess;
}

ncclResult_t nccl_ucx_isend(void *send_comm, void *data, int size, int tag, void *mhandle,
                            void **request) {
  ucx_send_comm_t *comm = (ucx_send_comm_t *)send_comm;
  ucx_mhandle_t   *mh   = (ucx_mhandle_t*)mhandle;
  ucx_request_t   *req;

  if (comm->ready == 0) {
    NCCLCHECK(ucx_send_check(comm));
    if (comm->ready == 0) {
      *request = NULL;
      return ncclSuccess;
    }
  }

#if UCP_API_VERSION >= UCP_VERSION(1, 10)
  ucp_request_param_t params;
  params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                        UCP_OP_ATTR_FIELD_DATATYPE;
  params.datatype     = ucp_dt_make_contig(1);
  params.cb.send      = send_handler_nbx;
  if (mh) {
    params.memory_type  = mh->mem_type;
    params.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMORY_TYPE;
  }
  req = ucp_tag_send_nbx(comm->ep, data, size, comm->tag, &params);
#else
  req = ucp_tag_send_nb(comm->ep, data, size, ucp_dt_make_contig(1),
                        comm->tag, send_handler);
#endif
  if (UCS_PTR_IS_ERR(req)) {
    WARN("ucx_isend: unable to send message (%s)", ucs_status_string(UCS_PTR_STATUS(req)));
    return ncclSystemError;
  } else if (req != NULL) {
    req->worker = comm->worker;
    req->size   = size;
  }
  *request = req ? req : (size ? (void*)REQUEST_COMPLETED_ZERO_LENGTGH:
                                 (void*)REQUEST_COMPLETED_NON_ZERO_LENGTH);

  return ncclSuccess;
}


ncclResult_t nccl_ucx_irecv(void *recv_comm, int n, void **data, int *sizes, int *tags, void **mhandle,
                            void **request) {
  ucx_recv_comm_t *comm = (ucx_recv_comm_t*)recv_comm;
  ucx_mhandle_t   *mh   = (ucx_mhandle_t*)mhandle[0];
  ucx_request_t   *req;

  if (comm->ready == 0) {
    NCCLCHECK(ucx_recv_check(comm));
    if (comm->ready == 0) {
      *request = NULL;
      return ncclSuccess;
    }
  }  

#if UCP_API_VERSION >= UCP_VERSION(1, 10)
  ucp_request_param_t params;
  params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                        UCP_OP_ATTR_FIELD_DATATYPE;
  params.datatype     = ucp_dt_make_contig(1);
  params.cb.recv      = recv_handler_nbx;
  if (mh) {
    params.memory_type  = mh->mem_type;
    params.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMORY_TYPE;
  }
  req = ucp_tag_recv_nbx(comm->worker, data[0], sizes[0], comm->tag, tag_mask, &params);
#else
  req = ucp_tag_recv_nb(comm->worker, data[0], size, ucp_dt_make_contig(1),
                        comm->tag, tag_mask, recv_handler);
#endif
  if (UCS_PTR_IS_ERR(req)) {
    WARN("ucx_irecv: unable to receive message (%s)", ucs_status_string(UCS_PTR_STATUS(req)));
    return ncclSystemError;
  } else if (req != NULL) {
    req->worker = comm->worker;
    req->size   = sizes[0];
  }
  *request = req ? req : (sizes[0] ? (void*)REQUEST_COMPLETED_ZERO_LENGTGH:
                                 (void*)REQUEST_COMPLETED_NON_ZERO_LENGTH);

  return ncclSuccess;
}

ncclResult_t nccl_ucx_iflush(void* recv_comm, int n, void** data, int* sizes, void** mhandle, void **request) {
  ucx_recv_comm_t *comm = (ucx_recv_comm_t *)recv_comm;
  ucx_mhandle_t   *mh   = (ucx_mhandle_t*)mhandle[0];
  ucx_request_t   *req;

  *request = NULL;
  int last = -1;
  for (int i=0; i<n; i++) if (sizes[i]) last = i;
  if (comm->gpuFlush.enabled == 0 || last == -1) return ncclSuccess;

  req = ucp_get_nb(comm->gpuFlush.flush_ep, &comm->gpuFlush.hostMem, 1,
                   (uint64_t)data[0], mh->rkey, send_handler);
  if (UCS_PTR_IS_ERR(req)) {
    WARN("ucx_flush: unable to read data (%s)", ucs_status_string(UCS_PTR_STATUS(req)));
    return ncclSystemError;
  } else if (req != NULL) {
    req->worker = comm->worker;
    req->size   = 1;
  }

  *request = req;
  return ncclSuccess;
}

ncclResult_t nccl_ucx_test(void *request, int *done, int *size) {
  ucx_request_t *req = (ucx_request_t *)request;
  unsigned      p    = 0;

  *done = 0;
  do {
    if (((uint64_t)request == REQUEST_COMPLETED_ZERO_LENGTGH) ||
        ((uint64_t)request == REQUEST_COMPLETED_NON_ZERO_LENGTH)) {
      *done = 1;
      if (size) {
        *size = -1 + (uint64_t)request;
      }
      return ncclSuccess;
    }

    if (req->completed == 1) {
      *done = 1;
      if (size) {
        *size = req->size;
      }
      req->completed = 0;
      ucp_request_free(req);
      return ncclSuccess;
    }
    p = ucp_worker_progress(req->worker);
  } while(p);

  return ncclSuccess;
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
  void *close_req;

  if (recv_comm){
    ucx_recv_comm_t *comm = (ucx_recv_comm_t*)recv_comm;

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

ncclNet_v6_t ucxPlugin_v6 = {
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
  .closeListen = nccl_ucx_close_listen
};

ncclNet_v5_t ucxPlugin_v5 = {
  .name = "UCX",
  .init = nccl_ucx_init,
  .devices = nccl_ucx_devices,
  .getProperties = nccl_ucx_get_properties,
  .listen = nccl_ucx_listen,
  .connect = nccl_ucx_connect,
  .accept = nccl_ucx_accept,
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

static ncclResult_t nccl_ucx_get_properties_v4(int dev, ncclNetProperties_v4_t* props) {
  ncclNetProperties_v6_t props_v6;
  ncclResult_t ret = nccl_ucx_get_properties(dev, &props_v6);
  if (ret != ncclSuccess) return ret;
  props->name = props_v6.name;
  props->pciPath = props_v6.pciPath;
  props->guid = props_v6.guid;
  props->ptrSupport = props_v6.ptrSupport;
  props->speed = props_v6.speed;
  props->port = props_v6.port;
  props->maxComms = props_v6.maxComms;
  return ncclSuccess;
};

static ncclResult_t nccl_ucx_isend_v4(void *sendComm, void* data, int size, void *mhandle, void** request) {
  return nccl_ucx_isend(sendComm, data, size, 0, mhandle, request);
}
static ncclResult_t nccl_ucx_irecv_v4(void* recvComm, void* data, int size, void* mhandle, void** request) {
  int tag = 0;
  return nccl_ucx_irecv(recvComm, 1, &data, &size, &tag, &mhandle, request);
}
static ncclResult_t nccl_ucx_iflush_v4(void* recvComm, void* data, int size, void* mhandle, void** request) {
  return nccl_ucx_iflush(recvComm, 1, &data, &size, &mhandle, request);
}
static ncclResult_t nccl_ucx_connect_v4(int dev, void* handle, void** sendComm) {
  ncclResult_t ret;
  do {
    ret = nccl_ucx_connect(dev, handle, sendComm);
  } while (ret == ncclSuccess && *sendComm == NULL);
  return ret;
}
static ncclResult_t nccl_ucx_accept_v4(void* listenComm, void** recvComm) {
  ncclResult_t ret;
  do {
    ret = nccl_ucx_accept(listenComm, recvComm);
  } while (ret == ncclSuccess && *recvComm == NULL);
  return ret;
}

ncclNet_v4_t ucxPlugin_v4 = {
  .name = "UCX",
  .init = nccl_ucx_init,
  .devices = nccl_ucx_devices,
  .getProperties = nccl_ucx_get_properties_v4,
  .listen = nccl_ucx_listen,
  .connect = nccl_ucx_connect_v4,
  .accept = nccl_ucx_accept_v4,
  .regMr = nccl_ucx_regmr,
  .deregMr = nccl_ucx_deregmr,
  .isend = nccl_ucx_isend_v4,
  .irecv = nccl_ucx_irecv_v4,
  .iflush = nccl_ucx_iflush_v4,
  .test = nccl_ucx_test,
  .closeSend = nccl_ucx_close_send,
  .closeRecv = nccl_ucx_close_recv,
  .closeListen = nccl_ucx_close_listen
};
