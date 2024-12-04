/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ucx_uct_lib.h"
#include "ucx_uct_ring.h"

#define NCCL_UCT_PENDING_SIZE 128
#define NCCL_UCT_PENDING_MASK (NCCL_UCT_PENDING_SIZE - 1)

/* Memory chunk to send or receive */
typedef struct {
  int                    tag;
  int                    size;
  void                   *data;
  union {
    uct_rkey_t           rkey;
    nccl_uct_memh_t      *uct_memh;
  } u;
  struct nccl_uct_rd_req *req;
  unsigned               index; /* irecv(): position in the receive request */
} nccl_uct_mem_t;

/* Context for GET requests to be posted */
typedef struct {
  uct_iov_t              iov;
  uint64_t               rva;
  uct_rkey_t             rkey;
  struct nccl_uct_rd_req *req;
} nccl_uct_get_param_t;

/* Communicator for client or server side */
typedef struct nccl_uct_rd_comm {
  /* Base communicator with endpoints setup */
  nccl_uct_comm_t        base;

  /* NCCL request free list */
  int                    req_count;
  struct nccl_uct_rd_req *free_req;

  /* TAG matching rings */
  nccl_uct_ring_t        exp;
  nccl_uct_ring_t        unexp;

  /* GET zcopy for matched chunks, but yet to be posted */
  struct {
    unsigned             first;
    unsigned             last;
    nccl_uct_get_param_t param[NCCL_UCT_PENDING_SIZE];
  } pending;
} nccl_uct_rd_comm_t;

/* Either irecv, isend or iflush NCCL request */
typedef struct nccl_uct_rd_req {
  uct_completion_t       completion; /* Release when count equals zero */
  int                    send_rts;   /* Request type */
  nccl_uct_rd_comm_t     *comm;      /* Parent communicator */
  struct nccl_uct_rd_req *next;      /* Free list node */

  int                    count;     /* isend(): 1, irecv(): from 1 to n */
  int                    rts_count; /* RTS actually received and matched */

  /* Sizes actually read to report, received from RTS */
  int                    sizes[NCCL_UCX_UCT_MAX_RECVS];

  /* Remote completed requests cookies, to send with ATS */
  struct nccl_uct_rd_req *remote_req[NCCL_UCX_UCT_MAX_RECVS];
} nccl_uct_rd_req_t;

static inline nccl_uct_rd_comm_t *
nccl_uct_rd_comm_get(nccl_uct_comm_t *base_comm) {
  return ucs_container_of(base_comm, nccl_uct_rd_comm_t, base);
}

static void nccl_uct_rd_send_ats(nccl_uct_rd_req_t *req) {
  ucs_status_t status;

  assert(req->send_rts == 0);
  assert(req->rts_count == req->count);
  assert(req->completion.count == 1);

  status = uct_ep_am_short(req->comm->base.uct_ep->ep, NCCL_UCT_AM_ATS,
                           (uint64_t)req->comm->base.remote.comm,
                           req->remote_req,
                           sizeof(*req->remote_req) * req->rts_count);
  if (status == UCS_OK) {
    req->completion.count--;
  }
}

static void nccl_uct_rd_pending_add(nccl_uct_rd_comm_t *comm,
                                    nccl_uct_mem_t *src, nccl_uct_mem_t *dst) {
  nccl_uct_rd_req_t *req = dst->req;
  nccl_uct_get_param_t *param;

  assert(src->size <= dst->size);
  assert(req->rts_count < NCCL_UCX_UCT_MAX_RECVS);

  req->sizes[dst->index]            = src->size;
  req->remote_req[req->rts_count++] = src->req; /* src->req is a cookie */

  if (src->size == 0) {
    req->completion.count--;
    return;
  }

  param = &comm->pending.param[comm->pending.last & NCCL_UCT_PENDING_MASK];
  comm->pending.last++;

  assert((comm->pending.first & NCCL_UCT_PENDING_MASK) !=
         (comm->pending.last & NCCL_UCT_PENDING_MASK));

  param->iov.buffer = dst->data;
  param->iov.length = src->size;
  param->iov.memh   = dst->u.uct_memh->memh;
  param->iov.stride = 0;
  param->iov.count  = 1;
  param->rva        = (uint64_t)src->data;
  param->rkey       = src->u.rkey;
  param->req        = req;
}

static void nccl_uct_rd_pending_drain(nccl_uct_rd_comm_t *comm) {
  ucs_status_t status;
  nccl_uct_get_param_t *param;

  for (; comm->pending.first != comm->pending.last; comm->pending.first++) {
    param = &comm->pending.param[comm->pending.first & NCCL_UCT_PENDING_MASK];

    status = uct_ep_get_zcopy(comm->base.uct_ep->ep, &param->iov, 1, param->rva,
                              param->rkey, &param->req->completion);
    if (status == UCS_OK) {
      param->req->completion.count--;
    } else if (status != UCS_INPROGRESS) {
      break;
    }

    if (param->req->completion.count == 1) {
      nccl_uct_rd_send_ats(param->req);
    }
  }
}

static ucs_status_t nccl_uct_rd_ats_callback(void *arg, void *data,
                                             size_t length, unsigned flags) {
  nccl_uct_rd_req_t **req  = (nccl_uct_rd_req_t **)((uint8_t *)data + 8);
  nccl_uct_rd_req_t **end  = (nccl_uct_rd_req_t **)((uint8_t *)data + length);

  for (; req + 1 <= end; req++) {
    assert((*req)->completion.count == 1);
    assert((*req)->comm == nccl_uct_rd_comm_get(*(nccl_uct_comm_t**)data));

    (*req)->completion.count = 0;
  }

  assert(req == end);
  return UCS_OK;
}

static ucs_status_t nccl_uct_rd_rts_callback(void *arg, void *data,
                                             size_t length, unsigned flags) {

  nccl_uct_rd_comm_t *comm = nccl_uct_rd_comm_get(*(nccl_uct_comm_t**)data);
  nccl_uct_mem_t *rts      = (nccl_uct_mem_t *)((uint8_t *)data + 8);
  nccl_uct_ring_t *exp;
  nccl_uct_mem_t *dst;
  unsigned i;

  assert(length == (sizeof(*rts) + 8));

  /* Do we already expect it? */
  exp = &comm->exp;
  i   = nccl_uct_ring_find(exp, rts->tag);
  if (i == exp->last) {
    nccl_uct_ring_append(&comm->unexp, rts->tag, rts, sizeof(*rts));
  } else {
    /* Receive request was already posted */
    dst = nccl_uct_ring_get_entry(exp, i);
    nccl_uct_rd_pending_add(comm, rts, dst);
    nccl_uct_ring_consume(exp, i);
  }

  return UCS_OK;
}

static ncclResult_t nccl_uct_rd_iface_set(nccl_uct_iface_t *uct_iface) {
  NCCLCHECK(nccl_uct_iface_set_handler(uct_iface, NCCL_UCT_AM_RTS,
                                       nccl_uct_rd_rts_callback));
  NCCLCHECK(nccl_uct_iface_set_handler(uct_iface, NCCL_UCT_AM_ATS,
                                       nccl_uct_rd_ats_callback));
  return ncclSuccess;
}

static ncclResult_t nccl_uct_rd_comm_alloc(nccl_uct_comm_t **comm_p) {
  nccl_uct_rd_comm_t *comm = calloc(1, sizeof(*comm));
  if (comm != NULL) {
    *comm_p = &comm->base;
    return ncclSuccess;
  }

  return ncclSystemError;
}

static ncclResult_t nccl_uct_rd_comm_init(nccl_uct_comm_t *base_comm,
                                          nccl_uct_context_t *context,
                                          nccl_uct_worker_t *worker, int dev,
                                          const nccl_uct_comm_t *remote_comm) {
  nccl_uct_rd_comm_t *comm = nccl_uct_rd_comm_get(base_comm);

  comm->pending.first = 0;
  comm->pending.last  = 0;
  comm->req_count     = 0;
  comm->free_req      = NULL;

  NCCLCHECK(nccl_uct_ring_init(&comm->exp, sizeof(nccl_uct_mem_t)));
  NCCLCHECK(nccl_uct_ring_init(&comm->unexp, sizeof(nccl_uct_mem_t)));

  return nccl_uct_comm_init(&comm->base, context, worker, dev, remote_comm);
}

static ncclResult_t nccl_uct_rd_init(ncclDebugLogger_t logFunction) {
  NCCL_STATIC_ASSERT(NCCL_UCT_RING_SIZE >= 2 * MAX_REQUESTS,
                     "Cannot handle expected/unexpected requests");
  NCCL_STATIC_ASSERT(NCCL_UCT_PENDING_SIZE > MAX_REQUESTS,
                     "Cannot handle enough pending requests");

  context.ops.comm_alloc = nccl_uct_rd_comm_alloc;
  context.ops.comm_init  = nccl_uct_rd_comm_init;
  context.ops.iface_set  = nccl_uct_rd_iface_set;
  context.rkey_size      = sizeof(((nccl_uct_mem_t*)0)->u.rkey);
  context.am_short_size  = sizeof(((nccl_uct_rd_req_t*)0)->remote_req);
  if (sizeof(nccl_uct_mem_t) > context.am_short_size) {
    context.am_short_size = sizeof(nccl_uct_mem_t);
  }

  return nccl_p2p_ib_init(&context.dev_count, &context.merge_dev_count, ncclIbDevs, context.if_name,
                          &context.if_addr, NULL, logFunction);
}

static nccl_uct_rd_req_t *nccl_uct_rd_req_alloc(nccl_uct_rd_comm_t *comm,
                                                int count) {
  nccl_uct_rd_req_t *req = comm->free_req;

  if (req == NULL) {
    req = malloc(sizeof(*req));
    if (req == NULL) {
      return req;
    }
  } else {
    comm->free_req = req->next;
  }

  comm->req_count++;
  req->comm              = comm;
  req->completion.func   = nccl_uct_empty_callback;
  req->completion.count  = count;
  req->completion.status = UCS_OK;
  return req;
}

static inline void nccl_uct_rd_req_free(nccl_uct_rd_req_t *req) {
  req->next           = req->comm->free_req;
  req->comm->free_req = req;
  req->comm->req_count--;
}

static ncclResult_t nccl_uct_rd_isend(void *send_comm, void *data, size_t size,
                                      int tag, void *mhandle, void **request) {

  nccl_uct_rd_comm_t *comm  = nccl_uct_rd_comm_get(send_comm);
  nccl_uct_memh_t *uct_memh = mhandle;
  nccl_uct_mem_t rts;
  nccl_uct_rd_req_t *req;
  ucs_status_t status;

  req = nccl_uct_rd_req_alloc(comm, 1);
  if (req == NULL) {
    *request = NULL;
    return ncclSuccess;
  }

  req->send_rts = 1;
  req->count    = 1;
  req->sizes[0] = size;
  *request      = req;

  rts.tag    = tag;
  rts.size   = size;
  rts.data   = data;
  rts.u.rkey = uct_memh->bundle.rkey;
  rts.req    = req;

  status = uct_ep_am_short(comm->base.uct_ep->ep, NCCL_UCT_AM_RTS,
                           (uint64_t)comm->base.remote.comm, &rts, sizeof(rts));
  if (status != UCS_OK) {
    nccl_uct_rd_req_free(req);
    *request = NULL;
  }

  return ncclSuccess;
}

static ncclResult_t nccl_uct_rd_isend_v8(void *send_comm, void *data, int size,
                                      int tag, void *mhandle, void **request) {
  return nccl_uct_rd_isend(send_comm, data, (size_t)size, tag, mhandle, request);
}

static ncclResult_t nccl_uct_rd_irecv(void *recv_comm, int n, void **data,
                                      size_t *sizes, int *tags, void **mhandles,
                                      void **request) {
  nccl_uct_rd_comm_t *comm   = nccl_uct_rd_comm_get(recv_comm);
  nccl_uct_memh_t **uct_memh = (nccl_uct_memh_t**)mhandles;
  nccl_uct_ring_t *unexp;
  nccl_uct_rd_req_t *req;
  nccl_uct_mem_t *rts, recv;
  unsigned i, j;

  assert(n <= NCCL_UCX_UCT_MAX_RECVS);

  /* Create a request */
  req      = nccl_uct_rd_req_alloc(comm, n + 1);
  *request = req;
  if (req == NULL) {
    return ncclSuccess;
  }

  req->send_rts  = 0;
  req->count     = n;
  req->rts_count = 0;

  /* Try to match or build expected list */
  for (i = 0; i < n; i++) {
    recv.tag        = tags[i];
    recv.size       = sizes[i];
    recv.data       = data[i];
    recv.u.uct_memh = uct_memh[i];
    recv.req        = req;
    recv.index      = i;

    unexp = &comm->unexp;
    j     = nccl_uct_ring_find(unexp, tags[i]);
    if (j == unexp->last) {
      nccl_uct_ring_append(&comm->exp, tags[i], &recv, sizeof(recv));
    } else {
      rts = nccl_uct_ring_get_entry(unexp, j);
      nccl_uct_rd_pending_add(comm, rts, &recv);
      nccl_uct_ring_consume(unexp, j);
    }
  }

  return ncclSuccess;
}

static ncclResult_t nccl_uct_rd_irecv_v8(void *recv_comm, int n, void **data,
                                      int *sizes, int *tags, void **mhandles,
                                      void **request) {
  size_t sizes_sizet[NCCL_NET_IB_MAX_RECVS];
  for (int i=0; i<n; i++) sizes_sizet[i] = sizes[i];
  return nccl_uct_rd_irecv(recv_comm, n, data, sizes_sizet, tags, mhandles, request);
}

static ncclResult_t nccl_uct_rd_iflush(void *recv_comm, int n, void **data,
                                       int *sizes, void **mhandle,
                                       void **request) {
  ncclResult_t result        = ncclSuccess;
  nccl_uct_comm_t *base_comm = recv_comm;
  nccl_uct_memh_t **uct_memh = (nccl_uct_memh_t**)mhandle;
  int last                   = nccl_uct_flush_index(base_comm, sizes, n);
  nccl_uct_rd_req_t *req;

  *request = NULL;

  if (last != -1) {
    req = nccl_uct_rd_req_alloc(nccl_uct_rd_comm_get(recv_comm), 1);
    if (req != NULL) {
      req->send_rts = -1;
      *request      = req;

      result = nccl_uct_flush(base_comm, data[last], sizes[last],
                              uct_memh[last], &req->completion, request);
      if (*request == NULL) {
        nccl_uct_rd_req_free(req);
      }
    }
  }

  return result;
}

static ncclResult_t nccl_uct_rd_test(void *request, int *done, int *sizes) {
  nccl_uct_rd_req_t *req = request;

  while (uct_worker_progress(req->comm->base.uct_worker->worker))
    ; /* empty */

  nccl_uct_rd_pending_drain(req->comm);

  if (req->completion.count > 0) {
    if ((req->send_rts == 0) && (req->completion.count == 1)) {
      nccl_uct_rd_send_ats(req);
    }

    if (req->completion.count > 0) {
      *done = 0;
      return ncclSuccess;
    }
  }

  if ((sizes != NULL) && (req->send_rts > -1)) {
    memcpy(sizes, req->sizes, req->count * sizeof(*req->sizes));
  }

  *done = 1;
  nccl_uct_rd_req_free(req);
  return ncclSuccess;
}

static ncclResult_t nccl_uct_rd_close(void *close_comm) {
  nccl_uct_rd_comm_t *comm = nccl_uct_rd_comm_get(close_comm);
  nccl_uct_rd_req_t *req;

  nccl_uct_comm_deinit(close_comm);

  while ((req = comm->free_req) != NULL) {
    comm->free_req = req->next;
    free(req);
  }

  assert(nccl_uct_ring_is_empty(&comm->exp));
  assert(nccl_uct_ring_is_empty(&comm->unexp));
  assert(comm->req_count == 0);
  assert(comm->pending.first == comm->pending.last);

  nccl_uct_ring_deinit(&comm->exp);
  nccl_uct_ring_deinit(&comm->unexp);
  free(comm);
  return ncclSuccess;
}

ncclNet_v9_t ucxUctRdPlugin_v9 = NCCL_UCT_PLUGIN_V9("UCX-UCT-RD", nccl_uct_rd);
ncclNet_v8_t ucxUctRdPlugin_v8 = NCCL_UCT_PLUGIN_V8("UCX-UCT-RD", nccl_uct_rd);
ncclNet_v7_t ucxUctRdPlugin_v7 = NCCL_UCT_PLUGIN_V7("UCX-UCT-RD", nccl_uct_rd);
ncclNet_v6_t ucxUctRdPlugin_v6 = NCCL_UCT_PLUGIN_V6("UCX-UCT-RD", nccl_uct_rd);
ncclNet_v5_t ucxUctRdPlugin_v5 = NCCL_UCT_PLUGIN_V5("UCX-UCT-RD", nccl_uct_rd);
