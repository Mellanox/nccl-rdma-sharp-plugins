/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ucx_uct_lib.h"

typedef enum {
  NCCL_UCT_AM_RTR = 14, /* Use particular values */
  NCCL_UCT_AM_ATP = 15
} nccl_uct_am_type_t;

typedef enum {
  NCCL_UCT_REQ_IRECV  = -1,
  NCCL_UCT_REQ_IFLUSH = -2
} nccl_uct_request_type_t;

struct nccl_uct_rdesc;

/* On-the-wire descriptor of a posted receive request entry */
typedef struct {
  int        tag;
  int        size;
  void       *data;
  int        matched;
  uct_rkey_t rkey;
} nccl_uct_chunk_t;

/* On-the-wire descriptor of a receive request containing many chunks */
typedef struct {
  uint64_t              id;
  uint16_t              count;
  uint32_t              size;
  struct nccl_uct_rdesc *peer_rdesc; /* Acts as a cookie along with id */
  nccl_uct_chunk_t      chunk[];
} nccl_uct_rdesc_hdr_t;

/* On-the-wire descriptor for receive request completion */
typedef struct {
  uint64_t              id;
  struct nccl_uct_rdesc *rdesc;
  int                   count; /* Number of sizes contained */
  int                   sizes[NCCL_UCX_UCT_MAX_RECVS];
} nccl_uct_atp_t;

/*
 * NCCL local request handler to progress:
 * - size -1 for multi receive
 * - size -2 for flush
 * - size > 0 for send
 */
typedef struct {
  /* Pending GET (iflush) PUT (isend) or receiving one ATP (irecv) */
  uct_completion_t      completion;
  int                   size;
  struct nccl_uct_rdesc *rdesc;
} nccl_uct_req_t;

/* Pending receive descriptor either on the receive or sending side */
typedef struct nccl_uct_rdesc {
  int                   nccl_usage; /* NCCL requests not finished/started */
  int                   send_atp;   /* >1 pending isend, ==1 pending atp send */

  union {
    ucs_list_link_t       list;  /* comm's linked list */
    struct nccl_uct_rdesc *next; /* inserted in free list */
  };

  struct nccl_uct_wr_comm  *comm;
  nccl_uct_rdesc_hdr_t     desc;
  nccl_uct_chunk_t         storage[NCCL_UCX_UCT_MAX_RECVS]; /* Don't use directly */
  nccl_uct_req_t           reqs[NCCL_UCX_UCT_MAX_RECVS];    /* NCCL requests */
  int                      sizes[NCCL_UCX_UCT_MAX_RECVS];   /* ATP received sizes */
} nccl_uct_rdesc_t;

typedef struct nccl_uct_wr_comm {
  nccl_uct_comm_t      base;

  int                  rdesc_alloc; /* Track allocated rdescs */
  nccl_uct_rdesc_t     *free_rdesc; /* Available rdesc for reuse */
  uint64_t             rdesc_id;    /* Next sequence number to use */

  /* Received RTRs: used by Sender communicator in ->isend() */
  ucs_list_link_t      rdesc_list;

} nccl_uct_wr_comm_t;

static inline nccl_uct_wr_comm_t *
nccl_uct_wr_comm_get(nccl_uct_comm_t *base_comm) {
  return ucs_container_of(base_comm, nccl_uct_wr_comm_t, base);
}

static nccl_uct_rdesc_t *nccl_uct_comm_rdesc_get(nccl_uct_wr_comm_t *comm) {
  nccl_uct_rdesc_t *rdesc = comm->free_rdesc;

  if (rdesc == NULL) {
    rdesc = calloc(1, sizeof(*rdesc));
  } else {
    comm->free_rdesc = rdesc->next;
  }

  rdesc->next = NULL;
  rdesc->comm = comm;
  comm->rdesc_alloc++;
  return rdesc;
}

static size_t nccl_uct_rdesc_size(int n) {
  return n * sizeof(nccl_uct_chunk_t) + sizeof(nccl_uct_rdesc_hdr_t);
}

/* Prepare a receive descriptor from irecv()/iflush() side */
static void nccl_uct_rdesc_set(nccl_uct_rdesc_t *rdesc, uint64_t id, int n,
                               void **data, int *sizes, int *tags,
                               nccl_uct_memh_t **uct_memh) {
  nccl_uct_rdesc_hdr_t *desc = &rdesc->desc;
  int i;

  /* Populate header */
  desc->id         = id;
  desc->count      = n;
  desc->size       = nccl_uct_rdesc_size(n);
  desc->peer_rdesc = rdesc; /* cookie, will be returned in ATP */

  /* Ref count that prevents NCCL from releasing memory */
  rdesc->nccl_usage = 1;
  rdesc->send_atp   = 0;

  /* Zero (iflush) or one or many receive request are contained */
  for (i = 0; i < n; i++) {
    desc->chunk[i].tag     = tags[i];
    desc->chunk[i].size    = sizes[i];
    desc->chunk[i].data    = data[i];
    desc->chunk[i].matched = 0;
    desc->chunk[i].rkey    = uct_memh[i]->bundle.rkey;
  }
}

static void nccl_uct_empty_callback(uct_completion_t *comp) {
  assert(comp->count == 0);
}

static nccl_uct_req_t *nccl_uct_rdesc_get_req(nccl_uct_rdesc_t *rdesc, int i,
                                              int size) {
  nccl_uct_req_t *req;

  assert(i < NCCL_UCX_UCT_MAX_RECVS);

  req        = &rdesc->reqs[i];
  req->size  = size;
  req->rdesc = rdesc;

  req->completion.func   = nccl_uct_empty_callback;
  req->completion.count  = 1;
  req->completion.status = UCS_OK;

  return &rdesc->reqs[i];
}

static void nccl_uct_comm_rdesc_put(nccl_uct_rdesc_t *rdesc) {
  nccl_uct_wr_comm_t *comm = rdesc->comm;

  assert(comm != NULL);

  rdesc->desc.id   = -1;
  rdesc->comm      = NULL;
  rdesc->next      = comm->free_rdesc;
  comm->free_rdesc = rdesc;
  comm->rdesc_alloc--;
}

/* On receiver side, after ->irecv(), expect corresponding ATP */
static ucs_status_t nccl_uct_atp_callback(void *arg, void *data, size_t length,
                                          unsigned flags) {
  nccl_uct_atp_t *atp = (nccl_uct_atp_t*)((uint8_t*)data + 8);

  assert(length == (sizeof(*atp) + 8));
  assert(*(nccl_uct_comm_t**)data == &atp->rdesc->comm->base);
  assert(atp->id == atp->rdesc->desc.id);
  assert(atp->count == atp->rdesc->desc.count);
  assert(atp->rdesc->reqs[0].completion.count == 1);

  atp->rdesc->reqs[0].completion.count--;
  memcpy(atp->rdesc->sizes, atp->sizes, atp->count * sizeof(*atp->sizes));
  return UCS_OK;
}

/* On sender side, asynchronously receive rdesc/RTR, later used by ->isend() */
static ucs_status_t nccl_uct_rtr_callback(void *arg, void *data, size_t length,
                                          unsigned flags) {
  nccl_uct_comm_t *base_comm = *(nccl_uct_comm_t **)data;
  nccl_uct_wr_comm_t *comm   = nccl_uct_wr_comm_get(base_comm);
  nccl_uct_rdesc_hdr_t *desc = (nccl_uct_rdesc_hdr_t*)((uint8_t*)data + 8);
  size_t size                = desc->size;
  nccl_uct_rdesc_t *rdesc;

  rdesc = nccl_uct_comm_rdesc_get(comm);
  if (rdesc == NULL) {
    WARN("Failed to get an rdesc in RTR callback");
    return UCS_ERR_NO_MEMORY; /* Cannot happend */
  }

  ucs_list_add_tail(&comm->rdesc_list, &rdesc->list);

  assert((size + 8) == length);
  assert(size == nccl_uct_rdesc_size(desc->count));

  memcpy(&rdesc->desc, desc, size);
  rdesc->nccl_usage = desc->count;
  rdesc->send_atp   = desc->count + 1;
  return UCS_OK;
}

static ncclResult_t nccl_uct_wr_iface_set(nccl_uct_iface_t *uct_iface) {
  NCCLCHECK(nccl_uct_iface_set_handler(uct_iface, NCCL_UCT_AM_RTR,
                                       nccl_uct_rtr_callback));
  NCCLCHECK(nccl_uct_iface_set_handler(uct_iface, NCCL_UCT_AM_ATP,
                                       nccl_uct_atp_callback));
  return ncclSuccess;
}

static ncclResult_t nccl_uct_wr_comm_alloc(nccl_uct_comm_t **comm_p) {
  nccl_uct_wr_comm_t *comm = calloc(1, sizeof(nccl_uct_wr_comm_t));
  if (comm != NULL) {
    *comm_p = &comm->base;
    return ncclSuccess;
  }

  return ncclSystemError;
}

static ncclResult_t nccl_uct_wr_comm_init(nccl_uct_comm_t *base_comm,
                                          nccl_uct_context_t *context,
                                          nccl_uct_worker_t *worker, int dev,
                                          const nccl_uct_comm_t *remote_comm) {
  nccl_uct_wr_comm_t *comm = nccl_uct_wr_comm_get(base_comm);

  ucs_list_head_init(&comm->rdesc_list);
  return nccl_uct_comm_init(&comm->base, context, worker, dev, remote_comm);
}

static ncclResult_t nccl_uct_wr_init(ncclDebugLogger_t logFunction) {
  context.ops.comm_alloc = nccl_uct_wr_comm_alloc;
  context.ops.comm_init  = nccl_uct_wr_comm_init;
  context.ops.iface_set  = nccl_uct_wr_iface_set;
  context.am_short_size  = nccl_uct_rdesc_size(NCCL_UCX_UCT_MAX_RECVS);
  context.rkey_size      = sizeof(((nccl_uct_chunk_t*)0)->rkey);

  return nccl_p2p_ib_init(&context.dev_count, ncclIbDevs, context.if_name,
                          &context.if_addr, NULL, logFunction);
}

/* Outcome is either send_atp equal to 1 or 0 */
static void nccl_uct_send_atp(nccl_uct_wr_comm_t *comm,
                              nccl_uct_rdesc_t *rdesc) {
  ucs_status_t status;
  nccl_uct_atp_t atp;
  int i;

  assert(rdesc->send_atp == 1);

  status = uct_ep_fence(comm->base.uct_ep->ep, 0);
  if (status != UCS_OK) {
    return;
  }

  atp.id    = rdesc->desc.id;
  atp.rdesc = rdesc->desc.peer_rdesc;
  atp.count = rdesc->desc.count;

  /* Sizes from isend() are lower or equal to their irecv() side */
  for (i = 0; i < rdesc->desc.count; i++) {
    atp.sizes[i] = rdesc->reqs[i].size;
  }

  status = uct_ep_am_short(comm->base.uct_ep->ep, NCCL_UCT_AM_ATP,
                           (uint64_t)comm->base.remote.comm, &atp, sizeof(atp));
  if (status == UCS_OK) {
    rdesc->send_atp = 0;
  }
}

static ncclResult_t nccl_uct_send(nccl_uct_wr_comm_t *comm, void *data,
                                  int size, nccl_uct_memh_t *uct_memh,
                                  nccl_uct_rdesc_t *rdesc, int i,
                                  void **request) {
  ucs_status_t status;
  uct_iov_t iov;
  nccl_uct_req_t *req;

  *request = NULL;

  /* Details for local data */
  iov.buffer = data;
  iov.length = size;
  iov.memh   = uct_memh->memh;
  iov.stride = iov.length;
  iov.count  = 1;

  assert(size <= rdesc->desc.chunk[i].size);

  req = nccl_uct_rdesc_get_req(rdesc, i, size); /* NCCL request */

  status = uct_ep_put_zcopy(comm->base.uct_ep->ep, &iov, 1,
                            (uint64_t)rdesc->desc.chunk[i].data,
                            rdesc->desc.chunk[i].rkey, &req->completion);

  if (status == UCS_OK) {
    req->completion.count--;
  } else if (status != UCS_INPROGRESS) {
    return ncclSuccess;
  }

  rdesc->desc.chunk[i].matched = 1;
  --rdesc->send_atp;

  if (rdesc->send_atp == 1) {
    ucs_list_del(&rdesc->list); /* all ->isend() were now matched */
    nccl_uct_send_atp(comm, rdesc);
  }

  *request = req;
  return ncclSuccess;
}

static ncclResult_t nccl_uct_wr_isend(void *send_comm, void *data, int size,
                                      int tag, void *mhandle, void **request) {
  nccl_uct_wr_comm_t *comm = nccl_uct_wr_comm_get(send_comm);
  nccl_uct_rdesc_t *rdesc;
  int i;

  *request = NULL;

  ucs_list_for_each(rdesc, &comm->rdesc_list, list) {
    for (i = 0; i < rdesc->desc.count; i++) {
      if (rdesc->desc.chunk[i].matched || (rdesc->desc.chunk[i].tag != tag)) {
        continue;
      }

      return nccl_uct_send(comm, data, size, mhandle, rdesc, i, request);
    }
  }

  /* Progress here to make sure we receive non-solicited RTRs */
  uct_worker_progress(comm->base.uct_worker->worker);
  return ncclSuccess;
}

static ncclResult_t nccl_uct_wr_irecv(void *recv_comm, int n, void **data,
                                      int *sizes, int *tags, void **mhandles,
                                      void **request) {
  nccl_uct_wr_comm_t *comm   = nccl_uct_wr_comm_get(recv_comm);
  nccl_uct_memh_t **uct_memh = (nccl_uct_memh_t**)mhandles;
  nccl_uct_rdesc_t *rdesc;
  ucs_status_t status;

  assert(n <= NCCL_UCX_UCT_MAX_RECVS);

  rdesc = nccl_uct_comm_rdesc_get(comm);
  if (rdesc == NULL) {
    return ncclInternalError;
  }

  nccl_uct_rdesc_set(rdesc, comm->rdesc_id++, n, data, sizes, tags, uct_memh);

  status = uct_ep_am_short(comm->base.uct_ep->ep, NCCL_UCT_AM_RTR,
                           (uint64_t)comm->base.remote.comm, &rdesc->desc,
                           nccl_uct_rdesc_size(n));
  if (status != UCS_OK) {
    nccl_uct_comm_rdesc_put(rdesc);
    *request = NULL;
  } else {
    /* Wait for receiving ATP */
    *request = nccl_uct_rdesc_get_req(rdesc, 0, NCCL_UCT_REQ_IRECV);
  }

  return ncclSuccess;
}

static ncclResult_t nccl_uct_wr_iflush(void *recv_comm, int n, void **data,
                                       int *sizes, void **mhandle,
                                       void **request) {
  nccl_uct_comm_t *base_comm = recv_comm;
  int last                   = nccl_uct_flush_index(base_comm, sizes, n);
  nccl_uct_memh_t **uct_memh = (nccl_uct_memh_t**)mhandle;
  nccl_uct_rdesc_t *rdesc;
  nccl_uct_req_t *req;
  ncclResult_t result;

  if (last == -1) {
    return ncclSuccess;
  }

  rdesc = nccl_uct_comm_rdesc_get(nccl_uct_wr_comm_get(base_comm));
  if (rdesc == NULL) {
    return ncclInternalError;
  }

  nccl_uct_rdesc_set(rdesc, ~0, 0, NULL, NULL, NULL, NULL);
  /* Wait for local GET completion */
  req      = nccl_uct_rdesc_get_req(rdesc, 0, NCCL_UCT_REQ_IFLUSH);
  *request = req;

  result = nccl_uct_flush(base_comm, data[last], sizes[last], uct_memh[last],
                          &req->completion, request);
  if (*request == NULL) {
    nccl_uct_comm_rdesc_put(rdesc);
  }

  return result;
}

static ncclResult_t nccl_uct_wr_test(void *request, int *done, int *sizes) {
  nccl_uct_req_t *req      = request;
  nccl_uct_rdesc_t *rdesc  = req->rdesc;
  nccl_uct_wr_comm_t *comm = rdesc->comm;

  uct_worker_progress(comm->base.uct_worker->worker);

  *done = 0;

  if (rdesc->send_atp == 1) {
    /* Slowpath */
    nccl_uct_send_atp(comm, rdesc);

    if (rdesc->send_atp && rdesc->nccl_usage == 1) {
      /* Keep the last isend request until ATP is out */
      return ncclSuccess;
    }
  }

  if (req->completion.count > 0) {
    return ncclSuccess;
  }

  *done = 1;

  if (req->size == NCCL_UCT_REQ_IRECV) {
    assert(&rdesc->reqs[0] == req);
    if (sizes != NULL) {
      memcpy(sizes, rdesc->sizes, rdesc->desc.count * sizeof(*sizes));
    }
  } else if (req->size == NCCL_UCT_REQ_IFLUSH) {
    assert(&rdesc->reqs[0] == req);
  } else {
    /* ->isend() request */
    assert(req->size > -1);
    if (sizes != NULL) {
      sizes[0] = req->size;
    }
  }

  if (--rdesc->nccl_usage < 1) {
    assert(rdesc->send_atp == 0);
    assert(rdesc->nccl_usage == 0);
    nccl_uct_comm_rdesc_put(rdesc);
  }

  return ncclSuccess;
}

static ncclResult_t nccl_uct_wr_close(void *close_comm) {
  nccl_uct_wr_comm_t *comm = nccl_uct_wr_comm_get(close_comm);
  nccl_uct_rdesc_t *rdesc;

  nccl_uct_comm_deinit(close_comm);

  while ((rdesc = comm->free_rdesc) != NULL) {
    comm->free_rdesc = rdesc->next;
    free(rdesc);
  }

  assert(ucs_list_is_empty(&comm->rdesc_list));
  assert(comm->rdesc_alloc == 0);
  free(comm);
  return ncclSuccess;
}

ncclNet_v8_t ucxUctPlugin_v8 = NCCL_UCT_PLUGIN_V8("UCX-UCT", nccl_uct_wr);
ncclNet_v7_t ucxUctPlugin_v7 = NCCL_UCT_PLUGIN_V7("UCX-UCT", nccl_uct_wr);
ncclNet_v6_t ucxUctPlugin_v6 = NCCL_UCT_PLUGIN_V6("UCX-UCT", nccl_uct_wr);
ncclNet_v5_t ucxUctPlugin_v5 = NCCL_UCT_PLUGIN_V5("UCX-UCT", nccl_uct_wr);
