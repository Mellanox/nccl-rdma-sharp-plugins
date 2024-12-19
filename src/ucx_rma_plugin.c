/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <pthread.h>

#include "nccl.h"
#include "net.h"
#include "p2p_plugin.h"

#include "ucp/api/ucp.h"

#define NCCL_UCP_HANDLE_MAGIC 0x3fea0433

#define UCXCHECK(cmd) do {                           \
  int e = cmd;                                       \
  if( UCS_OK != e ) {                                \
    WARN("Failed: UCX error %s:%d '%d' %s\n",        \
        __FILE__,__LINE__, e, ucs_status_string(e)); \
    return ncclInternalError;                        \
  }                                                  \
} while(0)

NCCL_PARAM(UCXAckDelay, "UCX_PUT_ACK_DELAY", 1);
NCCL_PARAM(UCXAckSkip, "UCX_PUT_ACK_SKIP", 0);

typedef enum {
  NCCL_UCP_TYPE_IRECV,
  NCCL_UCP_TYPE_ISEND,
  NCCL_UCP_TYPE_IFLUSH
} nccl_ucp_req_type_t;

/* Connection management state machine */
typedef enum {
  NCCL_UCP_START = 0,
  NCCL_UCP_CONNECT,
  NCCL_UCP_ACCEPT,
  NCCL_UCP_RECEIVE_REMOTE,
  NCCL_UCP_RX_READY,
  NCCL_UCP_DONE
} nccl_ucp_state_t;

typedef struct nccl_ucp_worker {
  struct nccl_ucp_worker *next;
  ucp_worker_h           ucp_worker;
  int                    dev;
  int                    used;
  ucp_context_h          ucp_context;
  void                   *address;
  size_t                 address_length;
} nccl_ucp_worker_t;

typedef struct {
  int                     dev_count;
  int                     listener_count;
  char                    if_name[MAX_IF_NAME_SIZE];
  union ncclSocketAddress if_addr;
  nccl_ucp_worker_t       *workers;
} nccl_ucp_context_t;

struct nccl_ucp_comm;

#define NCCL_UCP_RKEY_SIZE        96 /* bytes */
#define NCCL_UCP_WORKER_ADDR_SIZE 1024
#define NCCL_UCP_RKEY_COUNT       128 /* Maximum number of mh */
#define NCCL_UCP_MAX_RECV         8   /* Maximum chunks per ->irecv() */

/*
 * Max send request in-flight 8*8 = 64
 * Ring must be:  64+63 available slots
 */

#define NCCL_UCP_RING_SIZE 256
#define NCCL_UCP_RING_MASK (NCCL_UCP_RING_SIZE - 1)

#define REG_ALIGN (1 << 12) /* 4kB-pages */
#define REG_MASK  (REG_ALIGN - 1)

typedef struct nccl_ucp_packed {
  unsigned short rkey_id_start;
  int            rkey_buf_size;
  unsigned char  rkey_buf[NCCL_UCP_RKEY_SIZE];
  unsigned short rkey_id_end;
} __attribute__((aligned(64))) nccl_ucp_packed_rkey_t;

typedef struct nccl_ucp_chunk {
  uint64_t       data;
  int            size;
  int            tag;
  unsigned short rkey_id;
  unsigned short id;
} nccl_ucp_chunk_t;

typedef struct nccl_ucp_rtr {
  unsigned short   id_start; /* Id of the RTR */
  unsigned char    count;    /* Total chunks (at least 1) */
  char             avail;    /* Chunk left to proceed */
  char             ack;      /* Set if an ATP will be needed */
  nccl_ucp_chunk_t chunk[NCCL_UCP_MAX_RECV];
} __attribute__((aligned(64))) nccl_ucp_rtr_t;

struct nccl_ucp_comm;

typedef struct nccl_ucp_atp {
  unsigned short id_start; /* Id of the original RTR */
  unsigned char  count;    /* Added entries, incremented when posting */
  char           inflight; /* Chunk still being sent */
  char           reqs;     /* Count request alive */
  int            sizes[NCCL_UCP_MAX_RECV];
  unsigned short id; /* Id of the origin RTR again */
} __attribute__((aligned(64))) nccl_ucp_atp_t;

typedef struct nccl_ucp_share {
  nccl_ucp_packed_rkey_t packed_rkey[NCCL_UCP_RKEY_COUNT];
  nccl_ucp_rtr_t         rtr[NCCL_UCP_RING_SIZE];
  nccl_ucp_atp_t         atp[NCCL_UCP_RING_SIZE];
  unsigned               dummy_mem; /* Read-flush into it */
} nccl_ucp_share_t;

/* Exchanged OOB to connect to the remote communicator */
typedef struct nccl_ucp_address {
  /* Remote communicator pointer */
  struct nccl_ucp_comm *comm;

  /* Key and address for shared memory area */
  size_t               share_rkey_length;
  uint8_t              share_rkey[NCCL_UCP_RKEY_SIZE];
  nccl_ucp_share_t     *share;

  /* Worker address */
  size_t               address_length;
  uint8_t              address[NCCL_UCP_WORKER_ADDR_SIZE];
} nccl_ucp_address_t;

typedef struct {
  unsigned short rkey_id; /* Shared key identifier */
  int            mem_type;
  ucp_mem_h      ucp_memh;
  void           *rkey_buf; /* Packed key */
  size_t         rkey_buf_size;
  int            sent; /* Set to 1 only when PUT has been started */
  ucp_rkey_h     rkey; /* Set only for local read-based gpu flush */
} nccl_ucp_memh_t;

/* NCCL UCX plugin request */
typedef struct nccl_ucp_req {
  struct nccl_ucp_comm *comm;    /* Owner communicator */
  nccl_ucp_req_type_t  type;     /* Type of the request */
  unsigned short       rtr_id;   /* Id of the RTR received */
  int                  inflight; /* Set to zero when completed, irecv side */
} nccl_ucp_req_t;

/* Unpacked rkeys */
typedef struct {
  unsigned short rkey_id;
  ucp_rkey_h     rkey;
} nccl_ucp_rkey_t;

typedef struct nccl_ucp_comm {
  struct ncclSocket   sock;      /* OOB connection descriptor */
  int                 dev;       /* Device ID of the communicator */
  nccl_ucp_worker_t   *worker;   /* Worker for the communicator */
  int                 gpu_flush; /* True if enabled */
  nccl_ucp_req_type_t type;      /* Isend or Irecv side */

  unsigned short      req_id;  /* Next request id to use */
  unsigned short      rtr_id;  /* Next RTR id to use */
  unsigned short      rkey_id; /* Next rkey identifier */

  unsigned            total;         /* Current requests in progress */
  int                 inflight_rkey; /* Total remote keys being sent */
  int                 delay_atp;     /* Send ATP after remote completion */

  /* Connected endpoints */
  ucp_ep_h            ucp_ep;       /* Remote endpoint */
  ucp_ep_h            ucp_flush_ep; /* Local flush endpoint */

  /* In-flight NCCL-UCX requests (send/receive/flush) */
  nccl_ucp_req_t      req[NCCL_UCP_RING_SIZE];

  /* Unpacked received rkeys */
  nccl_ucp_rkey_t     rkey[NCCL_UCP_RKEY_COUNT];

  /* Local registered memory area */
  struct {
    nccl_ucp_share_t share;     /* Remotely accessible memory area */
    nccl_ucp_memh_t  *share_mh; /* Local memory handle of the share */
  } local;

  /* Remote shared memory area */
  struct {
    nccl_ucp_share_t *share;
    ucp_rkey_h       rkey;
  } remote;

  /* Remote worker address */
  nccl_ucp_address_t peer;
} nccl_ucp_comm_t;

typedef struct {
  nccl_ucp_state_t state;
  nccl_ucp_comm_t  *comm;

  int              offset;
  int              ready;
} nccl_ucp_stage_t;

typedef struct {
  int               dev;
  int               id;
  struct ncclSocket sock;
  nccl_ucp_stage_t  stage;
} nccl_ucp_listen_comm_t;

typedef struct {
  unsigned int magic;
  struct {
    int                     id;
    union ncclSocketAddress addr;
  } listener;
  nccl_ucp_stage_t stage;
} nccl_ucp_listen_handle_t;

static nccl_ucp_context_t context = {.dev_count = -1};

static pthread_mutex_t global_lock = PTHREAD_MUTEX_INITIALIZER;

static ncclResult_t nccl_ucx_rma_init(ncclDebugLogger_t logFunction) {
  return nccl_p2p_ib_init(&context.dev_count, ncclIbDevs, context.if_name,
                          &context.if_addr, NULL, logFunction);
}

static ncclResult_t nccl_ucx_rma_devices(int *ndev) {
  *ndev = context.dev_count;
  return ncclSuccess;
}

static ncclResult_t nccl_ucx_rma_get_properties(int dev,
                                                ncclNetProperties_t *props) {
  return nccl_p2p_ib_get_properties(ncclIbDevs, dev, props);
}

static ncclResult_t nccl_ucx_rma_listen(int dev, void *listen_handle,
                                        void **listen_comm) {
  nccl_ucp_listen_handle_t *handle = listen_handle;
  nccl_ucp_listen_comm_t *l_comm;
  union ncclSocketAddress addr;

  NCCL_STATIC_ASSERT(sizeof(nccl_ucp_listen_handle_t) < NCCL_NET_HANDLE_MAXSIZE,
                     "UCP listen handle is too big");

  l_comm = calloc(1, sizeof(*l_comm));
  if (l_comm == NULL) {
    return ncclSystemError;
  }

  /* Prepare socket */
  NCCLCHECK(ncclSocketInit(&l_comm->sock, &context.if_addr,
                           NCCL_UCP_HANDLE_MAGIC, ncclSocketTypeNetIb, NULL,
                           1));
  NCCLCHECK(ncclSocketListen(&l_comm->sock));
  NCCLCHECK(ncclSocketGetAddr(&l_comm->sock, &addr));

  /* Prepare listen communicator */
  l_comm->dev  = dev;
  l_comm->id   = context.listener_count++;
  *listen_comm = l_comm;

  /* Prepare handle to send */
  memset(handle, 0, sizeof(*handle));
  handle->magic         = NCCL_UCP_HANDLE_MAGIC;
  handle->listener.id   = l_comm->id;
  handle->listener.addr = addr;

  INFO(NCCL_INIT | NCCL_NET, "Listening id=%d dev=%d l_comm=%p", l_comm->id,
       dev, l_comm);
  return ncclSuccess;
}

static ncclResult_t nccl_ucx_rma_close_listen(void *listen_comm) {
  nccl_ucp_listen_comm_t *comm = listen_comm;

  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->sock));
    free(comm);
  }

  return ncclSuccess;
}

static ncclResult_t nccl_ucp_worker_init(nccl_ucp_worker_t *w, int dev,
                                         ucp_context_h ucp_context) {
  ucp_worker_params_t params = {.field_mask =
                                    UCP_WORKER_PARAM_FIELD_THREAD_MODE,
                                .thread_mode = UCS_THREAD_MODE_MULTI};
  ucp_worker_attr_t attr = {.field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE |
                                          UCP_WORKER_ATTR_FIELD_ADDRESS |
                                          UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS,
                            .address_flags = UCP_WORKER_ADDRESS_FLAG_NET_ONLY};

  w->dev          = dev;
  w->ucp_context  = ucp_context;
  w->next         = context.workers;
  context.workers = w;

  UCXCHECK(ucp_worker_create(w->ucp_context, &params, &w->ucp_worker));
  UCXCHECK(ucp_worker_query(w->ucp_worker, &attr));

  if (attr.thread_mode != UCS_THREAD_MODE_MULTI) {
    INFO(NCCL_NET, "Thread mode multi is not supported");
  }

  w->address_length = attr.address_length;
  w->address        = malloc(attr.address_length);
  if (w->address == NULL) {
    WARN("Failed to allocate worker address");
    goto err;
  }

  memcpy(w->address, attr.address, attr.address_length);
  ucp_worker_release_address(w->ucp_worker, attr.address);
  return ncclSuccess;

err:
  ucp_worker_release_address(w->ucp_worker, attr.address);
  ucp_worker_destroy(w->ucp_worker);
  return ncclSystemError;
}

static ncclResult_t nccl_ucp_context_create(int dev,
                                            ucp_context_h *ucp_context) {
  ucp_params_t params;
  ucp_config_t *config;
  char ucx_dev_name[128];
  ucs_status_t status;

  snprintf(ucx_dev_name, sizeof(ucx_dev_name), "%s:%d", ncclIbDevs[dev].devName,
           ncclIbDevs[dev].portNum);
  UCXCHECK(ucp_config_read("NCCL", NULL, &config));
  UCXCHECK(ucp_config_modify(config, "NET_DEVICES", ucx_dev_name));
  UCXCHECK(ucp_config_modify(config, "TLS", "rc_x"));

  params.field_mask = UCP_PARAM_FIELD_FEATURES;
  params.features   = UCP_FEATURE_RMA | UCP_FEATURE_AM;

  status = ucp_init(&params, config, ucp_context);
  ucp_config_release(config);
  NCCLCHECK(status);
  return ncclSuccess;
}

static nccl_ucp_worker_t *nccl_ucp_worker_get(int dev) {
  nccl_ucp_worker_t *w;
  ucp_context_h ucp_context;

  pthread_mutex_lock(&global_lock);
  w = calloc(1, sizeof(*w));
  if (w == NULL) {
    goto fail;
  }

  if (nccl_ucp_context_create(dev, &ucp_context) != ncclSuccess) {
    goto fail;
  }

  if (nccl_ucp_worker_init(w, dev, ucp_context) != ncclSuccess) {
    ucp_cleanup(ucp_context);
    goto fail;
  }

  w->used++;
  pthread_mutex_unlock(&global_lock);
  return w;

fail:
  free(w);
  pthread_mutex_unlock(&global_lock);
  return NULL;
}

static void nccl_ucp_worker_put(nccl_ucp_worker_t *worker) {
  int found = 0;
  nccl_ucp_worker_t **w;
  (void)found;

  pthread_mutex_lock(&global_lock);
  if (--worker->used < 1) {
    for (w = &context.workers; *w != NULL; w = &(*w)->next) {
      if (*w == worker) {
        *w    = worker->next;
        found = 1;
        break;
      }
    }

    assert(found == 1);
    assert(worker->used == 0);
    free(worker->address);
    ucp_worker_destroy(worker->ucp_worker);
    ucp_cleanup(worker->ucp_context);
    free(worker);
  }

  pthread_mutex_unlock(&global_lock);
}

static nccl_ucp_memh_t *nccl_ucp_mem_register(nccl_ucp_comm_t *comm, void *data,
                                              size_t size, int type) {
  uint64_t addr;
  nccl_ucp_memh_t *mh;
  ucp_mem_map_params_t params;
  ucs_status_t status;

  mh = calloc(1, sizeof(*mh));
  if (mh == NULL) {
    return NULL;
  }

  mh->mem_type =
      (type == NCCL_PTR_HOST) ? UCS_MEMORY_TYPE_HOST : UCS_MEMORY_TYPE_CUDA;
  addr = (uint64_t)data & ~REG_MASK;
  size = ROUNDUP(size + ((uint64_t)data & REG_MASK), REG_ALIGN);

  params.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                       UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                       UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
  params.address     = (void*)addr;
  params.length      = size;
  params.memory_type = mh->mem_type;

  status = ucp_mem_map(comm->worker->ucp_context, &params, &mh->ucp_memh);
  if (status != UCS_OK) {
    WARN("Memory registration failed for comm=%p mem=%p/%zu", comm, (void*)addr,
         size);
    free(mh);
    return NULL;
  }

  status = ucp_rkey_pack(comm->worker->ucp_context, mh->ucp_memh, &mh->rkey_buf,
                         &mh->rkey_buf_size);
  if (status != UCS_OK) {
    WARN("Rkey packing failed comm=%p", comm);
    ucp_mem_unmap(comm->worker->ucp_context, mh->ucp_memh);
    free(mh);
    return NULL;
  }

  return mh;
}

static ncclResult_t nccl_ucx_rma_deregmr(void *dereg_comm, void *mhandle) {
  nccl_ucp_comm_t *comm = dereg_comm;
  nccl_ucp_memh_t *mh   = mhandle;

  ucp_rkey_buffer_release(mh->rkey_buf);
  if (mh->rkey != NULL) {
    ucp_rkey_destroy(mh->rkey);
  }

  ucp_mem_unmap(comm->worker->ucp_context, mh->ucp_memh);
  free(mh);
  return ncclSuccess;
}

static ncclResult_t nccl_ucp_flush_ep_init(nccl_ucp_comm_t *comm) {
  ucp_worker_attr_t attr = {.field_mask    = UCP_WORKER_ATTR_FIELD_ADDRESS |
                                             UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS,
                            .address_flags = UCP_WORKER_ADDRESS_FLAG_NET_ONLY};
  ucp_ep_params_t params;

  UCXCHECK(ucp_worker_query(comm->worker->ucp_worker, &attr));

  params.field_mask =
      UCP_EP_PARAM_FIELD_REMOTE_ADDRESS | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
  params.address    = attr.address;
  params.err_mode = UCP_ERR_HANDLING_MODE_PEER; /* Mandatory with force close */
  UCXCHECK(
      ucp_ep_create(comm->worker->ucp_worker, &params, &comm->ucp_flush_ep));
  free(attr.address);
  return ncclSuccess;
}

static nccl_ucp_comm_t *nccl_ucp_comm_create(int dev,
                                             nccl_ucp_req_type_t type) {
  nccl_ucp_comm_t *comm = calloc(1, sizeof(*comm));
  if (comm == NULL) {
    return comm;
  }

  comm->worker = nccl_ucp_worker_get(dev);
  if (comm->worker == NULL) {
    goto err;
  }

  comm->local.share_mh = nccl_ucp_mem_register(
      comm, &comm->local.share, sizeof(comm->local.share), NCCL_PTR_HOST);
  if (comm->local.share_mh == NULL) {
    goto err;
  }

  comm->type      = type;
  comm->dev       = dev;
  comm->rtr_id    = 1;
  comm->req_id    = 1;
  comm->rkey_id   = 1;
  comm->delay_atp = !!ncclParamUCXAckDelay();
  comm->gpu_flush = (nccl_p2p_gdr_support(comm->dev) == ncclSuccess) ||
                    (nccl_p2p_dmabuf_support(comm->dev) == ncclSuccess);
  if (comm->gpu_flush && (nccl_ucp_flush_ep_init(comm) != ncclSuccess)) {
    nccl_ucx_rma_deregmr(comm, comm->local.share_mh);
    goto err;
  }

  return comm;

err:
  free(comm);
  return NULL;
}

static ncclResult_t nccl_ucp_ep_create(nccl_ucp_comm_t *comm) {
  ucp_ep_params_t params = {
    .field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                  UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE,
    .address    = (void*)comm->peer.address,
    .err_mode   = UCP_ERR_HANDLING_MODE_PEER /* Mandatory with force close */
  };

  UCXCHECK(ucp_ep_create(comm->worker->ucp_worker, &params, &comm->ucp_ep));
  UCXCHECK(ucp_ep_rkey_unpack(comm->ucp_ep, comm->peer.share_rkey,
                              &comm->remote.rkey));
  comm->remote.share = comm->peer.share;
  return ncclSuccess;
}

static ncclResult_t nccl_ucx_rma_address_send(nccl_ucp_comm_t *comm) {
  struct nccl_ucp_address peer;

  assert(comm->worker->address_length <= sizeof(peer.address));
  assert(comm->local.share_mh->rkey_buf_size <= sizeof(peer.share_rkey));

  peer.comm              = comm;
  peer.address_length    = comm->worker->address_length;
  peer.share_rkey_length = comm->local.share_mh->rkey_buf_size;
  peer.share             = &comm->local.share;
  memcpy(peer.address, comm->worker->address, comm->worker->address_length);
  memcpy(peer.share_rkey, comm->local.share_mh->rkey_buf,
         comm->local.share_mh->rkey_buf_size);

  return ncclSocketSend(&comm->sock, &peer, sizeof(peer));
}

static ncclResult_t nccl_ucx_rma_connect(int dev, void *listen_handle,
                                         void **send_comm,
                                         ncclNetDeviceHandle_t **sendDevComm) {
  nccl_ucp_listen_handle_t *handle = listen_handle;
  nccl_ucp_stage_t *stage          = &handle->stage;
  nccl_ucp_comm_t *comm            = stage->comm;
  int ready                        = 0;

  *send_comm = NULL;

  switch (stage->state) {
  case NCCL_UCP_START:
    comm        = nccl_ucp_comm_create(dev, NCCL_UCP_TYPE_ISEND);
    stage->comm = comm;
    if (stage->comm == NULL) {
      return ncclSystemError;
    }

    NCCLCHECK(ncclSocketInit(&stage->comm->sock, &handle->listener.addr,
                             handle->magic, ncclSocketTypeNetIb, NULL, 1));
    NCCLCHECK(ncclSocketConnect(&stage->comm->sock));

    stage->state = NCCL_UCP_CONNECT;
    /* fallthrough */

  case NCCL_UCP_CONNECT:
    NCCLCHECK(ncclSocketReady(&stage->comm->sock, &ready));
    if (!ready) {
      return ncclSuccess;
    }

    NCCLCHECK(nccl_ucx_rma_address_send(comm));

    stage->offset = 0;
    stage->state  = NCCL_UCP_RECEIVE_REMOTE;
    /* fallthrough */

  case NCCL_UCP_RECEIVE_REMOTE:
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->sock, &comm->peer,
                                 sizeof(comm->peer), &stage->offset));
    if (stage->offset != sizeof(comm->peer)) {
      return ncclSuccess;
    }

    NCCLCHECK(nccl_ucp_ep_create(comm));

    ready = 1;
    NCCLCHECK(ncclSocketSend(&comm->sock, &ready, sizeof(ready)));

    *send_comm    = comm;
    stage->ready  = 0;
    stage->offset = 0;
    stage->state  = NCCL_UCP_DONE;
    INFO(NCCL_INIT | NCCL_NET,
         "Connected comm=%p remote_comm=%p listener_id=%d "
         "ack_delay=%d ack_skip=%d",
         comm, comm->peer.comm, handle->listener.id, ncclParamUCXAckDelay(),
         ncclParamUCXAckSkip());
    break;

  default:
    break;
  }

  return ncclSuccess;
}

ncclResult_t nccl_ucx_rma_accept(void *listen_comm, void **recv_comm,
                                 ncclNetDeviceHandle_v7_t **recvDevComm) {
  nccl_ucp_listen_comm_t *l_comm = listen_comm;
  nccl_ucp_stage_t *stage        = &l_comm->stage;
  nccl_ucp_comm_t *comm          = stage->comm;
  int ready                      = 0;

  *recv_comm = NULL;

  switch (stage->state) {
  case NCCL_UCP_START:
    comm        = nccl_ucp_comm_create(l_comm->dev, NCCL_UCP_TYPE_IRECV);
    stage->comm = comm;
    if (stage->comm == NULL) {
      return ncclSystemError;
    }

    NCCLCHECK(ncclSocketInit(&comm->sock, NULL, NCCL_UCP_HANDLE_MAGIC,
                             ncclSocketTypeUnknown, NULL, 0));
    NCCLCHECK(ncclSocketAccept(&comm->sock, &l_comm->sock));

    stage->state = NCCL_UCP_ACCEPT;
    /* fallthrough */

  case NCCL_UCP_ACCEPT:
    NCCLCHECK(ncclSocketReady(&comm->sock, &ready));
    if (!ready) {
      return ncclSuccess;
    }

    stage->offset = 0;
    stage->state  = NCCL_UCP_RECEIVE_REMOTE;
    /* fallthrough */

  case NCCL_UCP_RECEIVE_REMOTE:
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->sock, &comm->peer,
                                 sizeof(comm->peer), &stage->offset));
    if (stage->offset != sizeof(comm->peer)) {
      return ncclSuccess;
    }

    NCCLCHECK(nccl_ucp_ep_create(comm));
    NCCLCHECK(nccl_ucx_rma_address_send(comm));

    stage->ready  = 0;
    stage->offset = 0;
    stage->state  = NCCL_UCP_RX_READY;
    /* fallthrough */

  case NCCL_UCP_RX_READY:
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->sock, &stage->ready,
                                 sizeof(stage->ready), &stage->offset));
    if (stage->offset != sizeof(stage->ready)) {
      return ncclSuccess; /* In progress */
    }

    assert(stage->ready == 1);
    *recv_comm   = comm;
    stage->state = NCCL_UCP_DONE;
    INFO(NCCL_INIT | NCCL_NET,
         "Accepted comm=%p peer_comm=%p listener_id=%d ack_delay=%d "
         "ack_skip=%d",
         comm, comm->peer.comm, l_comm->id, ncclParamUCXAckDelay(),
         ncclParamUCXAckSkip());
    break;

  default:
    break;
  }

  return ncclSuccess;
}

static void nccl_ucp_rdma_callback(void *request, ucs_status_t status,
                                   void *data) {
  int *inflight = data;
  assert(status == UCS_OK);
  assert(*inflight > 0);
  (*inflight)--;
  ucp_request_free(request);
}

static void nccl_ucp_rdma_isend_callback(void *request, ucs_status_t status,
                                         void *data) {
  nccl_ucp_req_t *req = data;

  nccl_ucp_rdma_callback(request, status, &req->inflight);
  req->comm->local.share.atp[req->rtr_id & NCCL_UCP_RING_MASK].inflight--;
}

static ucs_status_t nccl_ucp_shared_put(nccl_ucp_comm_t *comm, void *va,
                                        size_t size, void *rva, int *inflight) {
  ucp_request_param_t param = {};
  ucs_status_ptr_t status_ptr;

  assert((rva >= (void*)comm->remote.share) &&
         (rva + size) <=
             ((void*)comm->remote.share + sizeof(*comm->remote.share)));

  param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                       UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FIELD_MEMH |
                       UCP_OP_ATTR_FIELD_MEMORY_TYPE;
  param.cb.send      = nccl_ucp_rdma_callback;
  param.user_data    = inflight;
  param.memh         = comm->local.share_mh->ucp_memh;
  param.memory_type  = comm->local.share_mh->mem_type;

  status_ptr = ucp_put_nbx(comm->ucp_ep, va, size, (uint64_t)rva,
                           comm->remote.rkey, &param);
  return UCS_PTR_STATUS(status_ptr);
}

static ncclResult_t nccl_ucp_mh_update(nccl_ucp_comm_t *comm,
                                       nccl_ucp_memh_t *mh) {
  ucs_status_t status;
  nccl_ucp_packed_rkey_t *packed, *remote;

  if (!mh->sent) {
    packed = &comm->local.share.packed_rkey[mh->rkey_id];
    remote = &comm->remote.share->packed_rkey[mh->rkey_id];

    packed->rkey_buf_size = mh->rkey_buf_size;
    packed->rkey_id_start = mh->rkey_id;
    packed->rkey_id_end   = mh->rkey_id;
    memcpy(packed->rkey_buf, mh->rkey_buf, mh->rkey_buf_size);

    status = nccl_ucp_shared_put(comm, packed, sizeof(*packed), remote,
                                 &comm->inflight_rkey);
    if (UCS_STATUS_IS_ERR(status)) {
      WARN("Failed to send packed rkey");
      return ncclSystemError;
    }

    comm->inflight_rkey += (status == UCS_INPROGRESS);
    mh->sent             = 1;
  }

  return ncclSuccess;
}

static ncclResult_t nccl_ucx_rma_regmr(void *reg_comm, void *data, size_t size,
                                       int type, void **mhandle) {
  nccl_ucp_comm_t *comm = reg_comm;
  nccl_ucp_memh_t *mh;

  mh = nccl_ucp_mem_register(comm, data, size, type);
  if (mh) {
    mh->rkey_id = comm->rkey_id++;
    assert(mh->rkey_id < NCCL_UCP_RKEY_COUNT);

    if (comm->gpu_flush) {
      UCXCHECK(ucp_ep_rkey_unpack(comm->ucp_flush_ep, mh->rkey_buf, &mh->rkey));
    }
  }

  *mhandle = mh;
  return *mhandle ? ncclSuccess : ncclSystemError;
}

static ncclResult_t nccl_ucx_rma_regmr_dmabuf(void *comm, void *data,
                                              size_t size, int type,
                                              uint64_t offset, int fd,
                                              void **mhandle) {
  (void)fd; /* UCX performs the lookup automatically */
  assert(offset == 0);
  return nccl_ucx_rma_regmr(comm, data, size, type, mhandle);
}

static ncclResult_t nccl_ucx_rma_irecv(void *recv_comm, int n, void **data,
                                       int *sizes, int *tags, void **mhandle,
                                       void **request) {
  nccl_ucp_comm_t *comm = recv_comm;
  nccl_ucp_memh_t **mh  = (nccl_ucp_memh_t**)mhandle;
  nccl_ucp_req_t *req;
  nccl_ucp_rtr_t *rtr;
  nccl_ucp_atp_t *atp;
  int i;
  void *remote;
  ucs_status_t status;

  req = &comm->req[comm->req_id & NCCL_UCP_RING_MASK];
  rtr = &comm->local.share.rtr[comm->rtr_id & NCCL_UCP_RING_MASK];
  atp = &comm->local.share.atp[comm->rtr_id & NCCL_UCP_RING_MASK];

  assert(n <= NCCL_UCP_MAX_RECV);
  assert(req->comm == NULL);

  rtr->id_start = comm->rtr_id;
  rtr->count    = n;
  rtr->avail    = n;
  rtr->ack      = !((*request == (void*)0x1) && ncclParamUCXAckSkip());

  *request = NULL;

  for (i = 0; i < n; i++) {
    NCCLCHECK(nccl_ucp_mh_update(comm, mh[i]));

    rtr->chunk[i].data    = (uint64_t)data[i];
    rtr->chunk[i].rkey_id = mh[i]->rkey_id;
    rtr->chunk[i].size    = sizes[i];
    rtr->chunk[i].tag     = tags[i];
    rtr->chunk[i].id      = comm->rtr_id;
  }

  if (!rtr->ack) {
    atp->id_start = comm->rtr_id;
    atp->count    = n;
    atp->inflight = 0;
    atp->reqs     = 0;
    atp->id       = comm->rtr_id;
    memcpy(atp->sizes, sizes, sizeof(*sizes) * n);
  }

  remote = &comm->remote.share->rtr[comm->rtr_id & NCCL_UCP_RING_MASK];
  status = nccl_ucp_shared_put(
      comm, rtr, sizeof(*rtr) - (NCCL_UCP_MAX_RECV - n) * sizeof(*rtr->chunk),
      remote, &req->inflight);
  if (!UCS_STATUS_IS_ERR(status)) {
    req->comm     = comm;
    req->type     = NCCL_UCP_TYPE_IRECV;
    req->rtr_id   = comm->rtr_id;
    req->inflight = (status == UCS_INPROGRESS);

    comm->rtr_id++;
    comm->req_id++;
    comm->total++;

    *request = req;
  }

  return ncclSuccess;
}

static ucp_rkey_h nccl_ucp_rkey_get(nccl_ucp_comm_t *comm,
                                    unsigned short rkey_id) {
  nccl_ucp_rkey_t *nccl_rkey;
  nccl_ucp_packed_rkey_t *packed;
  ucs_status_t status;

  assert(rkey_id < NCCL_UCP_RKEY_COUNT);
  nccl_rkey = &comm->rkey[rkey_id];
  if (nccl_rkey->rkey_id != rkey_id) {
    /* Try to unpack */
    __sync_synchronize();
    packed = &comm->local.share.packed_rkey[rkey_id];
    if ((packed->rkey_id_start != rkey_id) ||
        (packed->rkey_id_end != rkey_id)) {
      return NULL;
    }

    status =
        ucp_ep_rkey_unpack(comm->ucp_ep, packed->rkey_buf, &nccl_rkey->rkey);
    if (status != UCS_OK) {
      return NULL;
    }

    nccl_rkey->rkey_id = rkey_id;
  }

  return nccl_rkey->rkey;
}

static ncclResult_t nccl_ucp_send(nccl_ucp_comm_t *comm, unsigned short id,
                                  unsigned i, void *data, int size,
                                  nccl_ucp_memh_t *mh, void **request) {
  nccl_ucp_req_t *req;
  nccl_ucp_rtr_t *rtr;
  nccl_ucp_atp_t *atp;
  ucs_status_ptr_t status_ptr;
  ucp_request_param_t param;
  ucp_rkey_h rkey;

  *request = NULL;
  atp      = &comm->local.share.atp[id & NCCL_UCP_RING_MASK];
  rtr      = &comm->local.share.rtr[id & NCCL_UCP_RING_MASK];
  req      = &comm->req[comm->req_id & NCCL_UCP_RING_MASK];
  assert(req->comm == NULL);
  assert(rtr->avail > 0);
  assert(rtr->id_start == id);

  rkey = nccl_ucp_rkey_get(comm, rtr->chunk[i].rkey_id);
  if (rkey == NULL) {
    return ncclSuccess;
  }

  param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                       UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FIELD_MEMH |
                       UCP_OP_ATTR_FIELD_MEMORY_TYPE;
  param.cb.send      = nccl_ucp_rdma_isend_callback;
  param.user_data    = req;
  param.memh         = mh->ucp_memh;
  param.memory_type  = mh->mem_type;

  status_ptr =
      ucp_put_nbx(comm->ucp_ep, data, size, rtr->chunk[i].data, rkey, &param);
  if (UCS_PTR_IS_ERR(status_ptr)) {
    return ncclSuccess;
  }

  if (rtr->avail == rtr->count) {
    assert(atp->reqs == 0);
    assert(atp->inflight == 0);
    atp->id_start = rtr->id_start;
    atp->count    = 0;
    memset(atp->sizes, 0, sizeof(atp->sizes));
    atp->id = rtr->id_start;
  }

  req->comm      = comm;
  req->type      = NCCL_UCP_TYPE_ISEND;
  req->rtr_id    = rtr->id_start;
  req->inflight  = UCS_PTR_IS_PTR(status_ptr);
  atp->inflight += req->inflight;
  atp->sizes[i]  = size;
  atp->count++;
  atp->reqs++;

  rtr->avail--;
  rtr->chunk[i].tag = INT_MAX;

  comm->req_id++;
  comm->total++;
  *request = req;
  return ncclSuccess;
}

static ncclResult_t nccl_ucx_rma_isend(void *send_comm, void *data, int size,
                                       int tag, void *mhandle, void **request) {
  ncclResult_t result   = ncclSuccess;
  nccl_ucp_comm_t *comm = send_comm;
  volatile nccl_ucp_rtr_t *rtr;
  unsigned short id;
  unsigned i;

  *request = NULL;

  assert(tag != INT_MAX);
  for (id = comm->rtr_id;; id++) {
    rtr = &comm->local.share.rtr[id & NCCL_UCP_RING_MASK];
    if ((rtr->id_start != id) || (rtr->chunk->id != id)) {
      break;
    }

    for (i = 0; i < rtr->count; i++) {
      while (rtr->chunk[i].id != id) {
        __sync_synchronize();
      }
    }

    if (rtr->avail < 1) {
      if (id == comm->rtr_id) {
        comm->rtr_id++;
      }
      continue;
    }

    for (i = 0; i < rtr->count; i++) {
      if (rtr->chunk[i].tag == tag) {
        result = nccl_ucp_send(comm, id, i, data, size, mhandle, request);
        goto out;
      }
    }
  }

out:
  if ((*request == NULL) && (comm->total == 0)) {
    ucp_worker_progress(comm->worker->ucp_worker);
  }

  return result;
}

static int nccl_ucp_flush_index(nccl_ucp_comm_t *comm, int *sizes, int n) {
  int i, last = -1;

  if (comm->gpu_flush) {
    for (i = 0; i < n; i++) {
      if (sizes[i]) {
        last = i;
      }
    }
  }

  return last;
}

static ncclResult_t nccl_ucx_rma_iflush(void *recv_comm, int n, void **data,
                                        int *sizes, void **mhandle,
                                        void **request) {
  nccl_ucp_comm_t *comm = recv_comm;
  nccl_ucp_memh_t **mh  = (nccl_ucp_memh_t**)mhandle;
  int last              = nccl_ucp_flush_index(comm, sizes, n);
  nccl_ucp_req_t *req;
  ucs_status_ptr_t status_ptr;
  ucp_request_param_t param;

  if (last == -1) {
    *request = NULL;
    return ncclSuccess;
  }

  req                = &comm->req[comm->req_id & NCCL_UCP_RING_MASK];
  param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                       UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FIELD_MEMH |
                       UCP_OP_ATTR_FIELD_MEMORY_TYPE;
  param.cb.send      = nccl_ucp_rdma_callback;
  param.user_data    = &req->inflight;
  param.memh         = comm->local.share_mh->ucp_memh;
  param.memory_type  = UCS_MEMORY_TYPE_HOST;

  status_ptr = ucp_get_nbx(comm->ucp_flush_ep, &comm->local.share.dummy_mem, 1,
                           (uint64_t)data[last], mh[last]->rkey, &param);
  assert(!UCS_PTR_IS_ERR(status_ptr));
  assert(req->comm == NULL);

  req->type     = NCCL_UCP_TYPE_IFLUSH;
  req->inflight = (UCS_PTR_STATUS(status_ptr) == UCS_INPROGRESS);
  req->comm     = comm;

  comm->req_id++;
  comm->total++;
  *request = req;
  return ncclSuccess;
}

static void nccl_ucx_rma_close_ep(ucp_ep_h ep) {
  void *req;
  ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS,
                               .flags        = UCP_EP_CLOSE_FLAG_FORCE};

  req = ucp_ep_close_nbx(ep, &param);
  (void)req;
  assert(req == NULL);
}

static ncclResult_t nccl_ucx_rma_close_comm(void *close_comm) {
  int i;
  nccl_ucp_comm_t *comm = close_comm;

  assert(comm->total == 0);
  assert(comm->inflight_rkey == 0);

  nccl_ucx_rma_close_ep(comm->ucp_ep);
  if (comm->ucp_flush_ep != NULL) {
    assert(comm->gpu_flush);
    nccl_ucx_rma_close_ep(comm->ucp_flush_ep);
  }

  for (i = 0; i < NCCL_UCP_RKEY_COUNT; i++) {
    if (comm->rkey[i].rkey != NULL) {
      ucp_rkey_destroy(comm->rkey[i].rkey);
    }
  }

  for (i = 0; i < NCCL_UCP_RING_SIZE; i++) {
    if (comm->type == NCCL_UCP_TYPE_ISEND) {
      assert(comm->local.share.rtr[i].avail < 1);
      assert(comm->local.share.atp[i].reqs == 0);
      assert(comm->local.share.atp[i].inflight == 0);
    }

    assert(comm->req[i].comm == NULL);
  }

  if (comm->remote.rkey != NULL) {
    ucp_rkey_destroy(comm->remote.rkey);
  }

  if (comm->local.share_mh != NULL) {
    nccl_ucx_rma_deregmr(comm, comm->local.share_mh);
  }
  ncclSocketClose(&comm->sock);
  nccl_ucp_worker_put(comm->worker);
  free(comm);
  return ncclSuccess;
}

static void nccl_ucp_req_release(nccl_ucp_req_t *req) {
  assert(req->comm->total > 0);
  req->comm->total--;
  req->comm = NULL;
}

static ncclResult_t nccl_ucx_rma_test(void *request, int *done, int *sizes) {
  nccl_ucp_req_t *req   = request;
  nccl_ucp_comm_t *comm = req->comm;
  nccl_ucp_atp_t *atp;
  nccl_ucp_rtr_t *rtr;
  ucs_status_t status;
  void *remote;

  *done = 0;
  while (ucp_worker_progress(comm->worker->ucp_worker) != 0)
    ; /* nothing */

  if (req->type == NCCL_UCP_TYPE_ISEND) {
    rtr    = &comm->local.share.rtr[req->rtr_id & NCCL_UCP_RING_MASK];
    atp    = &comm->local.share.atp[req->rtr_id & NCCL_UCP_RING_MASK];
    remote = &comm->remote.share->atp[req->rtr_id & NCCL_UCP_RING_MASK];

    assert(comm->type == NCCL_UCP_TYPE_ISEND);
    assert(rtr->id_start == req->rtr_id);
    assert(atp->id_start == req->rtr_id);

    if (rtr->avail == 0) {
      if (rtr->ack) {
        if (atp->inflight &&
            (comm->delay_atp ||
             (ucp_worker_fence(comm->worker->ucp_worker) != UCS_OK))) {
          return ncclSuccess;
        }

        status         = nccl_ucp_shared_put(comm, atp, sizeof(*atp), remote,
                                             &req->inflight);
        req->inflight += (status == UCS_INPROGRESS);
        rtr->avail    -= !UCS_STATUS_IS_ERR(status);
      } else {
        rtr->avail--;
      }
    }

    *done = (req->inflight == 0) && ((atp->reqs > 1) || (rtr->avail < 0));
    if (*done) {
      atp->reqs--;
      assert((atp->reqs > 0) || (atp->inflight == 0));
      nccl_ucp_req_release(req);
    }
  } else if (req->type == NCCL_UCP_TYPE_IRECV) {
    assert(comm->type == NCCL_UCP_TYPE_IRECV);
    atp = &comm->local.share.atp[req->rtr_id & NCCL_UCP_RING_MASK];
    __sync_synchronize();
    *done = (req->inflight == 0) && (atp->id_start == req->rtr_id) &&
            (atp->id == req->rtr_id) &&
            ((comm->total > 1) || (comm->inflight_rkey == 0));
    if (*done) {
      if (sizes != NULL) {
        memcpy(sizes, atp->sizes, sizeof(*atp->sizes) * atp->count);
      }
      nccl_ucp_req_release(req);
    }
  } else {
    assert(req->type == NCCL_UCP_TYPE_IFLUSH);
    assert(comm->type == NCCL_UCP_TYPE_IRECV);
    *done = (req->inflight == 0) &&
            ((comm->total > 1) || (comm->inflight_rkey == 0));
    if (*done) {
      nccl_ucp_req_release(req);
    }
  }

  return ncclSuccess;
}

static ncclResult_t nccl_ucx_rma_regmr_v7(void *comm, void *data, int size,
                                          int type, void **mhandle) {
  return nccl_ucx_rma_regmr(comm, data, (size_t)size, type, mhandle);
}

static ncclResult_t
nccl_ucx_rma_get_properties_v7(int dev, ncclNetProperties_v7_t *props_v7) {
  ncclNetProperties_t props;
  ncclResult_t ret = nccl_ucx_rma_get_properties(dev, &props);
  if (ret != ncclSuccess) {
    return ret;
  }
  props_v7->name             = props.name;
  props_v7->pciPath          = props.pciPath;
  props_v7->guid             = props.guid;
  props_v7->ptrSupport       = props.ptrSupport;
  props_v7->speed            = props.speed;
  props_v7->latency          = props.latency;
  props_v7->port             = props.port;
  props_v7->maxComms         = props.maxComms;
  props_v7->maxRecvs         = props.maxRecvs;
  props_v7->netDeviceType    = props.netDeviceType;
  props_v7->netDeviceVersion = props.netDeviceVersion;
  return ncclSuccess;
}

static ncclResult_t
nccl_ucx_rma_get_properties_v6(int dev, ncclNetProperties_v6_t *props_v6) {
  ncclNetProperties_t props;
  ncclResult_t ret = nccl_ucx_rma_get_properties(dev, &props);
  if (ret != ncclSuccess) {
    return ret;
  }
  props_v6->name       = props.name;
  props_v6->pciPath    = props.pciPath;
  props_v6->guid       = props.guid;
  props_v6->ptrSupport = props.ptrSupport;
  props_v6->speed      = props.speed;
  props_v6->latency    = props.latency;
  props_v6->port       = props.port;
  props_v6->maxComms   = props.maxComms;
  props_v6->maxRecvs   = props.maxRecvs;
  return ncclSuccess;
}

static ncclResult_t nccl_ucx_rma_connect_v6(int dev, void *handle,
                                            void **send_comm) {
  ncclNetDeviceHandle_v7_t *dev_handle = NULL;
  return nccl_ucx_rma_connect(dev, handle, send_comm, &dev_handle);
}

static ncclResult_t nccl_ucx_rma_accept_v6(void *listen_comm,
                                           void **recv_comm) {
  ncclNetDeviceHandle_v7_t *dev_handle = NULL;
  return nccl_ucx_rma_accept(listen_comm, recv_comm, &dev_handle);
}

#define UCX_RMA_PLUGIN_NAME "UCX-RMA"
ncclNet_v8_t ucxRmaPlugin_v8 = {
  .name          = UCX_RMA_PLUGIN_NAME,
  .init          = nccl_ucx_rma_init,
  .devices       = nccl_ucx_rma_devices,
  .getProperties = nccl_ucx_rma_get_properties,
  .listen        = nccl_ucx_rma_listen,
  .connect       = nccl_ucx_rma_connect,
  .accept        = nccl_ucx_rma_accept,
  .regMr         = nccl_ucx_rma_regmr,
  .regMrDmaBuf   = nccl_ucx_rma_regmr_dmabuf,
  .deregMr       = nccl_ucx_rma_deregmr,
  .isend         = nccl_ucx_rma_isend,
  .irecv         = nccl_ucx_rma_irecv,
  .iflush        = nccl_ucx_rma_iflush,
  .test          = nccl_ucx_rma_test,
  .closeSend     = nccl_ucx_rma_close_comm,
  .closeRecv     = nccl_ucx_rma_close_comm,
  .closeListen   = nccl_ucx_rma_close_listen,
};

ncclNet_v7_t ucxRmaPlugin_v7 = {
  .name          = UCX_RMA_PLUGIN_NAME,
  .init          = nccl_ucx_rma_init,
  .devices       = nccl_ucx_rma_devices,
  .getProperties = nccl_ucx_rma_get_properties_v7,
  .listen        = nccl_ucx_rma_listen,
  .connect       = nccl_ucx_rma_connect,
  .accept        = nccl_ucx_rma_accept,
  .regMr         = nccl_ucx_rma_regmr_v7,
  .regMrDmaBuf   = nccl_ucx_rma_regmr_dmabuf,
  .deregMr       = nccl_ucx_rma_deregmr,
  .isend         = nccl_ucx_rma_isend,
  .irecv         = nccl_ucx_rma_irecv,
  .iflush        = nccl_ucx_rma_iflush,
  .test          = nccl_ucx_rma_test,
  .closeSend     = nccl_ucx_rma_close_comm,
  .closeRecv     = nccl_ucx_rma_close_comm,
  .closeListen   = nccl_ucx_rma_close_listen,
};

ncclNet_v6_t ucxRmaPlugin_v6 = {
    .name          = UCX_RMA_PLUGIN_NAME,
    .init          = nccl_ucx_rma_init,
    .devices       = nccl_ucx_rma_devices,
    .getProperties = nccl_ucx_rma_get_properties_v6,
    .listen        = nccl_ucx_rma_listen,
    .connect       = nccl_ucx_rma_connect_v6,
    .accept        = nccl_ucx_rma_accept_v6,
    .regMr         = nccl_ucx_rma_regmr_v7,
    .regMrDmaBuf   = nccl_ucx_rma_regmr_dmabuf,
    .deregMr       = nccl_ucx_rma_deregmr,
    .isend         = nccl_ucx_rma_isend,
    .irecv         = nccl_ucx_rma_irecv,
    .iflush        = nccl_ucx_rma_iflush,
    .test          = nccl_ucx_rma_test,
    .closeSend     = nccl_ucx_rma_close_comm,
    .closeRecv     = nccl_ucx_rma_close_comm,
    .closeListen   = nccl_ucx_rma_close_listen
};

ncclNet_v5_t ucxRmaPlugin_v5 = {
    .name          = UCX_RMA_PLUGIN_NAME,
    .init          = nccl_ucx_rma_init,
    .devices       = nccl_ucx_rma_devices,
    .getProperties = nccl_ucx_rma_get_properties_v6,
    .listen        = nccl_ucx_rma_listen,
    .connect       = nccl_ucx_rma_connect_v6,
    .accept        = nccl_ucx_rma_accept_v6,
    .regMr         = nccl_ucx_rma_regmr_v7,
    .deregMr       = nccl_ucx_rma_deregmr,
    .isend         = nccl_ucx_rma_isend,
    .irecv         = nccl_ucx_rma_irecv,
    .iflush        = nccl_ucx_rma_iflush,
    .test          = nccl_ucx_rma_test,
    .closeSend     = nccl_ucx_rma_close_comm,
    .closeRecv     = nccl_ucx_rma_close_comm,
    .closeListen   = nccl_ucx_rma_close_listen
};
