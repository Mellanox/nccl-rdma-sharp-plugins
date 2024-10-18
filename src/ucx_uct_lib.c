/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ucx_uct_lib.h"


static pthread_mutex_t nccl_uct_lock = PTHREAD_MUTEX_INITIALIZER;

nccl_uct_context_t context = {
    .tl_name   = "rc_mlx5",
    .dev_count = -1,
    .merge_dev_count = -1
};

void nccl_uct_empty_callback(uct_completion_t *comp) {
  assert(comp->count == 0);
}

ncclResult_t nccl_uct_iface_set_handler(nccl_uct_iface_t *uct_iface,
                                        int id, uct_am_callback_t callback) {
  UCXCHECK(uct_iface_set_am_handler(uct_iface->iface, id, callback, NULL, 0),
           return ncclInternalError, "get AM handler id=%d", id);
  return ncclSuccess;
}

static const uct_device_addr_t *
nccl_uct_ep_addr_dev(const nccl_uct_ep_addr_t *addr) {
  return (uct_device_addr_t*)addr->data;
}

static const uct_ep_addr_t *
nccl_uct_ep_addr_ep(const nccl_uct_ep_addr_t *addr) {
  return (uct_ep_addr_t*)(addr->data + addr->dev_addr_size);
}

static uct_iface_h nccl_uct_resource_iface_open(uct_worker_h worker,
                                                uct_md_h md,
                                                uct_tl_resource_desc_t *tl) {
  uct_iface_params_t params = {};
  ucs_status_t status;
  uct_iface_config_t *config;
  uct_iface_h iface;

  UCXCHECK(uct_md_iface_config_read(md, tl->tl_name, NULL, NULL, &config),
           return NULL, "read MD iface config for TL '%s'", tl->tl_name);

  status = uct_config_modify(config, "IB_TX_INLINE_RESP", "0");
  if (status != UCS_OK) {
      WARN("Failed to modify MD configuration for '%s', error %s",
           tl->tl_name, ucs_status_string(status));
      uct_config_release(config);
      return NULL;
  }

  params.field_mask =
      UCT_IFACE_PARAM_FIELD_OPEN_MODE | UCT_IFACE_PARAM_FIELD_DEVICE |
      UCT_IFACE_PARAM_FIELD_STATS_ROOT | UCT_IFACE_PARAM_FIELD_RX_HEADROOM;
  params.open_mode            = UCT_IFACE_OPEN_MODE_DEVICE;
  params.mode.device.tl_name  = tl->tl_name;
  params.mode.device.dev_name = tl->dev_name;
  params.stats_root           = NULL;
  params.rx_headroom          = 0;

  status = uct_iface_open(md, worker, &params, config, &iface);
  uct_config_release(config);
  UCXCHECK(status, return NULL, "open UCT iface %s/%s",
                      tl->tl_name, tl->dev_name);

  uct_iface_progress_enable(iface, UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);

  return iface;
}

static uct_iface_h
nccl_uct_md_iface_open(uct_worker_h worker, uct_component_h comp,
                       unsigned md_index, const char *md_name,
                       const char *tl_name, const char *dev_name,
                       uct_md_h *md_p) {
  uct_iface_h iface = NULL;
  ucs_status_t status;
  uct_md_config_t *md_config;
  uct_md_h md;
  uct_md_attr_t md_attr;
  uct_tl_resource_desc_t *tls;
  unsigned tls_count, i;

  UCXCHECK(uct_md_config_read(comp, NULL, NULL, &md_config), return NULL,
           "read MD[%d] config", md_index);

  status = uct_md_open(comp, md_name, md_config, &md);
  uct_config_release(md_config);
  UCXCHECK(status, return NULL, "open MD[%d/%s]", md_index, md_name);

  UCXCHECK(uct_md_query(md, &md_attr), goto out, "query MD[%d/%s]", md_index,
           md_name);

  UCXCHECK(uct_md_query_tl_resources(md, &tls, &tls_count), goto out,
           "query resources MD[%d/%s]", md_index, md_name);

  for (i = 0; i < tls_count; i++) {
    if (!strcmp(dev_name, tls[i].dev_name) &&
        !strcmp(tl_name, tls[i].tl_name)) {

      iface = nccl_uct_resource_iface_open(worker, md, &tls[i]);
      break;
    }
  }

  uct_release_tl_resource_list(tls);

out:
  if (iface == NULL) {
    uct_md_close(md);
  } else {
    *md_p = md;
  }
  return iface;
}

static nccl_uct_iface_t *nccl_uct_iface_open(nccl_uct_worker_t *uct_worker,
                                             const char *tl_name,
                                             const char *dev_name) {
  uct_worker_h worker         = uct_worker->worker;
  nccl_uct_iface_t *uct_iface = NULL;
  uct_iface_h iface           = NULL;
  uct_component_h *comps, *comp;
  unsigned comps_count, i;
  uct_component_attr_t comp_attr;
  uct_iface_attr_t iface_attr;
  uct_md_h md;
  uct_md_attr_t md_attr;

  UCXCHECK(uct_query_components(&comps, &comps_count), return NULL,
           "query component list");

  for (comp = comps; comp < comps + comps_count; comp++) {
    comp_attr.field_mask = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT;
    UCXCHECK(uct_component_query(*comp, &comp_attr), goto out,
             "query component");

    comp_attr.field_mask = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES;
    comp_attr.md_resources =
        alloca(sizeof(*comp_attr.md_resources) * comp_attr.md_resource_count);
    UCXCHECK(uct_component_query(*comp, &comp_attr), goto out,
             "query component resources");

    for (i = 0; i < comp_attr.md_resource_count; i++) {
      iface = nccl_uct_md_iface_open(worker, *comp, i,
                                     comp_attr.md_resources[i].md_name, tl_name,
                                     dev_name, &md);
      if (iface != NULL) {
        goto found;
      }
    }
  }

  if (iface == NULL) {
    WARN("Failed to open iface for tl_name=%s dev_name=%s", tl_name, dev_name);
    goto out;
  }

found:
  UCXCHECK(uct_iface_query(iface, &iface_attr), goto fail,
           "iface for tl_name=%s dev_name=%s", tl_name, dev_name);

  if (!(iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP)) {
    WARN("Interface flag CONNECT_TO_EP is not set");
    goto fail;
  }

  if (!(iface_attr.cap.flags & UCT_IFACE_FLAG_GET_ZCOPY) ||
      !(iface_attr.cap.flags & UCT_IFACE_FLAG_PUT_ZCOPY)) {
    WARN("Interface does not support ZCOPY (flags=0x%x)", iface_attr.cap.flags);
    goto fail;
  }

  UCXCHECK(uct_md_query(md, &md_attr), goto fail, "query md for iface %p",
           iface);

  uct_iface = calloc(1, sizeof(*uct_iface));
  if (uct_iface == NULL) {
    WARN("Failed to alloc uct iface structure");
    goto fail;
  }

  uct_iface->ep_addr_size     = iface_attr.ep_addr_len;
  uct_iface->md               = md;
  uct_iface->comp             = *comp;
  uct_iface->rkey_packed_size = md_attr.rkey_packed_size;

  if (iface_attr.cap.flags & UCT_IFACE_FLAG_AM_SHORT) {
    uct_iface->am_max_short = iface_attr.cap.am.max_short;
  }

  if (iface_attr.cap.flags & UCT_IFACE_FLAG_GET_ZCOPY) {
    uct_iface->min_get_zcopy = iface_attr.cap.get.min_zcopy;
  }

  if (context.am_short_size > uct_iface->am_max_short) {
    WARN("Failed RTR does not fit in face AM short (%dB > %dB)",
         context.am_short_size, uct_iface->am_max_short);
    goto fail;
  }

  if (uct_iface->rkey_packed_size > context.rkey_size) {
    WARN("Interface rkey_packed_size %d too big", uct_iface->rkey_packed_size);
    goto fail;
  }

  if (iface_attr.device_addr_len > 0) {
    uct_iface->dev_addr_size = iface_attr.device_addr_len;
    uct_iface->dev_addr      = calloc(1, iface_attr.device_addr_len);
    if (uct_iface->dev_addr == NULL) {
      WARN("Failed to alloc dev_addr");
      goto fail;
    }

    UCXCHECK(uct_iface_get_device_address(iface, uct_iface->dev_addr),
             goto fail, "query iface device addr for tl_name=%s dev_name=%s",
             tl_name, dev_name);
  }

  if (iface_attr.iface_addr_len > 0) {
    uct_iface->addr_size = iface_attr.iface_addr_len;
    uct_iface->addr      = calloc(1, iface_attr.iface_addr_len);
    if (uct_iface->addr == NULL) {
      WARN("Failed to alloc iface addr");
      goto fail;
    }

    UCXCHECK(uct_iface_get_address(iface, uct_iface->addr), goto fail,
             "query iface addr to tl_name=%s dev_name=%s", tl_name, dev_name);
  }

  uct_iface->iface = iface;

out:
  uct_release_component_list(comps);
  return uct_iface;

fail:
  if (uct_iface != NULL) {
    free(uct_iface->dev_addr);
    free(uct_iface->addr);
    free(uct_iface);
  }
  if (iface != NULL) {
    uct_iface_close(iface);
  }
  uct_release_component_list(comps);
  return NULL;
}

static const char *nccl_dev_name(int dev) {
  static __thread char buf[128];
  snprintf(buf, sizeof(buf), "%s:%d", ncclIbDevs[dev].devName,
           ncclIbDevs[dev].portNum);
  return buf;
}

static void nccl_uct_iface_close(nccl_uct_iface_t *uct_iface) {
  uct_iface_close(uct_iface->iface);
  uct_md_close(uct_iface->md);
  free(uct_iface->dev_addr);
  free(uct_iface->addr);
  free(uct_iface);
}

static nccl_uct_worker_t *nccl_uct_worker_create(nccl_uct_context_t *context,
                                                 int dev) {
  nccl_uct_worker_t *w = NULL;
  ncclResult_t result;

  w = calloc(1, sizeof(*w));
  if (w == NULL) {
    WARN("Failed worker allocation: dev=%d", dev);
    return NULL;
  }

  UCXCHECK(ucs_async_context_create(UCS_ASYNC_MODE_THREAD_SPINLOCK, &w->async),
           goto fail_free, "create UCT async context: dev=%d", dev);
  UCXCHECK(uct_worker_create(w->async, UCS_THREAD_MODE_SINGLE, &w->worker),
           goto fail_context, "create UCT worker: dev=%d", dev);

  w->id.dev    = dev;
  w->id.thread = pthread_self();
  w->context   = context;

  w->uct_iface = nccl_uct_iface_open(w, context->tl_name, nccl_dev_name(dev));
  if (w->uct_iface == NULL) {
    WARN("Failed to create UCT iface for worker: dev=%d", dev);
    goto fail;
  }

  result = context->ops.iface_set(w->uct_iface);
  if (result != ncclSuccess) {
    WARN("Failed to set iface configuration: dev=%d", dev);
    goto fail;
  }

  w->next              = context->worker_list;
  context->worker_list = w;
  return w;

fail:
  if (w->uct_iface != NULL) {
    nccl_uct_iface_close(w->uct_iface);
  }

  uct_worker_destroy(w->worker);
fail_context:
  ucs_async_context_destroy(w->async);
fail_free:
  free(w);
  return NULL;
}

static nccl_uct_worker_t *nccl_uct_worker_get(nccl_uct_context_t *context,
                                              int dev) {
  nccl_uct_worker_t *w;

  pthread_mutex_lock(&nccl_uct_lock);
  for (w = context->worker_list; w != NULL; w = w->next) {
    if (w->id.dev == dev) {
      goto found;
    }
  }

  w = nccl_uct_worker_create(context, dev);
  if (w == NULL) {
    goto out;
  }

found:
  w->count++;
out:
  pthread_mutex_unlock(&nccl_uct_lock);
  return w;
}

static void nccl_uct_ep_destroy(nccl_uct_ep_t *uct_ep) {
  uct_ep_destroy(uct_ep->ep);
  free(uct_ep);
}

static nccl_uct_ep_t *nccl_uct_ep_create(nccl_uct_iface_t *uct_iface) {
  nccl_uct_ep_t *uct_ep;
  uct_ep_params_t ep_params;

  uct_ep = calloc(1, sizeof(*uct_ep) + uct_iface->ep_addr_size);
  if (uct_ep == NULL) {
    WARN("Failed to alloc EP memory");
    return NULL;
  }

  uct_ep->addr      = (uct_ep_addr_t*)uct_ep->data;
  uct_ep->uct_iface = uct_iface;

  ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
  ep_params.iface      = uct_iface->iface;

  UCXCHECK(uct_ep_create(&ep_params, &uct_ep->ep), goto fail, "create UCT EP");
  UCXCHECK(uct_ep_get_address(uct_ep->ep, uct_ep->addr), goto fail_destroy,
           "get UCT EP address");

  return uct_ep;

fail_destroy:
  nccl_uct_ep_destroy(uct_ep);
fail:
  free(uct_ep);
  return NULL;
}

ncclResult_t nccl_uct_comm_init(nccl_uct_comm_t *comm,
                                nccl_uct_context_t *context,
                                nccl_uct_worker_t *worker, int dev,
                                const nccl_uct_comm_t *remote_comm) {
  if (worker == NULL) {
    worker = nccl_uct_worker_get(context, dev);
  }

  comm->uct_worker = worker;
  if (comm->uct_worker == NULL) {
    return ncclSystemError;
  }

  comm->dev         = dev;
  comm->context     = context;
  comm->remote.comm = remote_comm;
  comm->uct_iface   = comm->uct_worker->uct_iface;
  comm->uct_ep      = nccl_uct_ep_create(comm->uct_iface);
  if (comm->uct_ep == NULL) {
    return ncclSystemError;
  }

  return ncclSuccess;
}

static ncclResult_t nccl_uct_ep_addr_set(nccl_uct_ep_addr_t *addr,
                                         const nccl_uct_comm_t *comm,
                                         const nccl_uct_ep_t *uct_ep) {
  nccl_uct_iface_t *uct_iface = comm->uct_iface;
  size_t total = uct_iface->dev_addr_size + uct_iface->ep_addr_size;

  if (total > sizeof(addr->data)) {
    WARN("Address sizes are too big (%zu + %u > %zu)", uct_iface->dev_addr_size,
         uct_iface->ep_addr_size);
    return ncclSystemError;
  }

  addr->dev_addr_size = uct_iface->dev_addr_size;
  addr->ep_addr_size  = uct_iface->ep_addr_size;

  memcpy(addr->data, uct_iface->dev_addr, addr->dev_addr_size);
  memcpy(addr->data + addr->dev_addr_size, uct_ep->addr,
         uct_iface->ep_addr_size);
  return ncclSuccess;
}

static ncclResult_t nccl_uct_ep_connect_to_ep(nccl_uct_ep_t *uct_ep,
                                              nccl_uct_ep_addr_t *addr) {
  UCXCHECK(uct_ep_connect_to_ep(uct_ep->ep, nccl_uct_ep_addr_dev(addr),
                                nccl_uct_ep_addr_ep(addr)),
           return ncclSystemError, "connect to EP");

  return ncclSuccess;
}

ncclResult_t nccl_uct_devices(int *ndev) {
  *ndev = context.dev_count;
  return ncclSuccess;
}

ncclResult_t nccl_uct_listen(int dev, void *listen_handle, void **listen_comm) {
  nccl_uct_listen_handle_t *handle = listen_handle;
  nccl_uct_listen_comm_t *l_comm   = calloc(1, sizeof(*l_comm));
  nccl_uct_comm_t *accept_comm;
  union ncclSocketAddress addr;

  if (l_comm == NULL) {
    WARN("Failed to alloc UCT listener(dev=%d)", dev);
    return ncclSystemError;
  }

  NCCL_STATIC_ASSERT(sizeof(nccl_uct_listen_handle_t) < NCCL_NET_HANDLE_MAXSIZE,
                     "UCT listen handle is too big");

  NCCLCHECK(ncclSocketInit(&l_comm->sock, &context.if_addr,
                           NCCL_UCT_LISTEN_HANDLE_MAGIC, ncclSocketTypeNetIb,
                           NULL, 1, 0));
  NCCLCHECK(ncclSocketListen(&l_comm->sock));
  NCCLCHECK(ncclSocketGetAddr(&l_comm->sock, &addr));

  l_comm->uct_worker = nccl_uct_worker_get(&context, dev);
  if (l_comm->uct_worker == NULL) {
    WARN("Failed to create worker for listener dev=%d", dev);
    return ncclSystemError;
  }

  NCCLCHECK(context.ops.comm_alloc(&accept_comm));

  l_comm->comm    = accept_comm;
  l_comm->context = &context;
  l_comm->dev     = dev;
  l_comm->id      = context.listener_count++;

  *listen_comm = l_comm;

  memset(handle, 0, sizeof(*handle));
  handle->magic         = NCCL_UCT_LISTEN_HANDLE_MAGIC;
  handle->listener.id   = l_comm->id;
  handle->listener.addr = addr;
  handle->comm          = accept_comm;

  INFO(NCCL_INIT | NCCL_NET, "Listening id=%d dev=%d comm=%p", l_comm->id, dev,
       handle->comm);
  return ncclSuccess;
}

ncclResult_t nccl_uct_close_listen(void *listen_comm) {
  nccl_uct_listen_comm_t *comm = listen_comm;

  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->sock));
    free(comm);
  }
  return ncclSuccess;
}

static ncclResult_t nccl_uct_comm_gpu_flush_init(nccl_uct_comm_t *comm) {
  size_t size = comm->uct_iface->min_get_zcopy;

  comm->gpu_flush.enabled = (nccl_p2p_gdr_support(comm->dev) == ncclSuccess) ||
                            (nccl_p2p_dmabuf_support(comm->dev) == ncclSuccess);

  if (!comm->gpu_flush.enabled) {
    return ncclSuccess;
  }

  comm->gpu_flush.mem = malloc(size);
  if (comm->gpu_flush.mem == NULL) {
    goto fail;
  }

  comm->gpu_flush.uct_ep = nccl_uct_ep_create(comm->uct_iface);
  if (comm->gpu_flush.uct_ep == NULL) {
    goto fail_free_mem;
  }

  NCCLCHECK(nccl_uct_ep_addr_set(&comm->gpu_flush.addr, comm,
                                 comm->gpu_flush.uct_ep));
  NCCLCHECK(
      nccl_uct_ep_connect_to_ep(comm->gpu_flush.uct_ep, &comm->gpu_flush.addr));
  UCXCHECK(uct_md_mem_reg(comm->uct_iface->md, (void*)comm->gpu_flush.mem, size,
                          UCT_MD_MEM_ACCESS_ALL, &comm->gpu_flush.memh),
           goto fail_destroy_ep, "GPU flush memory registration");

  return ncclSuccess;

fail_destroy_ep:
  nccl_uct_ep_destroy(comm->gpu_flush.uct_ep);
fail_free_mem:
  free(comm->gpu_flush.mem);
fail:
  comm->gpu_flush.enabled = 0;
  return ncclSystemError;
}

ncclResult_t nccl_uct_connect(int dev, void *listen_handle, void **send_comm,
                              ncclNetDeviceHandle_t **sendDevComm) {
  int ready                        = 0;
  nccl_uct_listen_handle_t *handle = listen_handle;
  nccl_uct_stage_t *stage          = &handle->stage;
  nccl_uct_comm_t *comm            = stage->comm;
  struct nccl_uct_comm_remote remote;

  *send_comm = NULL;

  switch (stage->state) {
  case NCCL_UCT_START:
    NCCLCHECK(context.ops.comm_alloc(&comm));
    NCCLCHECK(context.ops.comm_init(comm, &context, NULL, dev, handle->comm));
    NCCLCHECK(ncclSocketInit(&comm->sock, &handle->listener.addr, handle->magic,
                             ncclSocketTypeNetIb, NULL, 1, 0));
    NCCLCHECK(ncclSocketConnect(&comm->sock));

    stage->comm  = comm;
    stage->state = NCCL_UCT_CONNECT;
    /* fallthrough */

  case NCCL_UCT_CONNECT:
    NCCLCHECK(ncclSocketReady(&comm->sock, &ready));
    if (!ready) {
      return ncclSuccess;
    }

    NCCLCHECK(nccl_uct_ep_addr_set(&remote.addr.rma, comm, comm->uct_ep));
    remote.comm = comm;
    NCCLCHECK(ncclSocketSend(&comm->sock, &remote, sizeof(remote)));

    stage->offset = 0;
    stage->state  = NCCL_UCT_RECEIVE_ADDR;
    /* fallthrough */

  case NCCL_UCT_RECEIVE_ADDR:
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->sock,
                                 &comm->remote.addr, sizeof(comm->remote.addr),
                                 &stage->offset, NULL));
    if (stage->offset != sizeof(comm->remote.addr)) {
      return ncclSuccess; /* In progress */
    }

    ready = 1;
    NCCLCHECK(nccl_uct_ep_connect_to_ep(comm->uct_ep, &comm->remote.addr.rma));
    NCCLCHECK(ncclSocketSend(&comm->sock, &ready, sizeof(ready)));

    *send_comm   = comm;
    stage->state = NCCL_UCT_DONE;
    INFO(NCCL_INIT | NCCL_NET,
         "Connected comm=%p remote_comm=%p listener_id=%d", comm,
         comm->remote.comm, handle->listener.id);
    break;

  default:
    WARN("UCT connnect for dev=%d using unsupported state %d", dev,
         stage->state);
    return ncclSystemError;
  }

  return ncclSuccess;
}

ncclResult_t nccl_uct_accept(void *listen_comm, void **recv_comm,
                             ncclNetDeviceHandle_v7_t **recvDevComm) {
  nccl_uct_listen_comm_t *l_comm = listen_comm;
  nccl_uct_stage_t *stage        = &l_comm->stage;
  nccl_uct_comm_t *comm          = stage->comm;
  nccl_uct_comm_addr_t addr;
  int ready;

  *recv_comm = NULL;

  switch (stage->state) {
  case NCCL_UCT_START:
    comm = l_comm->comm;

    NCCLCHECK(ncclSocketInit(&comm->sock, NULL, NCCL_SOCKET_MAGIC,
                             ncclSocketTypeUnknown, NULL, 0, 0));
    NCCLCHECK(ncclSocketAccept(&comm->sock, &l_comm->sock));
    NCCLCHECK(context.ops.comm_init(comm, l_comm->context, l_comm->uct_worker,
                                    l_comm->dev, NULL));
    NCCLCHECK(nccl_uct_comm_gpu_flush_init(comm));

    stage->comm  = comm;
    stage->state = NCCL_UCT_ACCEPT;
    /* fallthrough */

  case NCCL_UCT_ACCEPT:
    NCCLCHECK(ncclSocketReady(&comm->sock, &ready));
    if (!ready) {
      return ncclSuccess;
    }

    NCCLCHECK(nccl_uct_ep_addr_set(&addr.rma, comm, comm->uct_ep));
    NCCLCHECK(ncclSocketSend(&comm->sock, &addr, sizeof(addr)));

    stage->offset = 0;
    stage->state  = NCCL_UCT_RECEIVE_REMOTE;
    /* fallthrough */

  case NCCL_UCT_RECEIVE_REMOTE:
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->sock, &comm->remote,
                                 sizeof(comm->remote), &stage->offset, NULL));
    if (stage->offset != sizeof(comm->remote)) {
      return ncclSuccess;
    }

    NCCLCHECK(nccl_uct_ep_connect_to_ep(comm->uct_ep, &comm->remote.addr.rma));

    stage->ready  = 0;
    stage->offset = 0;
    stage->state  = NCCL_UCT_RX_READY;
    /* fallthrough */

  case NCCL_UCT_RX_READY:
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->sock, &stage->ready,
                                 sizeof(stage->ready), &stage->offset, NULL));
    if (stage->offset != sizeof(ready)) {
      return ncclSuccess;
    }
    if (stage->ready != 1) {
      WARN("Accepted comm=%p invalid status (ready=%d)", comm, stage->ready);
      return ncclSystemError;
    }

    *recv_comm   = comm;
    stage->state = NCCL_UCT_DONE;
    INFO(NCCL_INIT | NCCL_NET, "Accepted comm=%p remote_comm=%p listener_id=%d",
         comm, comm->remote.comm, l_comm->id);
    break;

  default:
    WARN("UCT accept for dev=%d using unsupported state %d", l_comm->dev,
         stage->state);
    return ncclSystemError;
  }

  return ncclSuccess;
}

ncclResult_t nccl_uct_reg_mr(void *reg_comm, void *data, size_t size, int type,
                             void **mhandle) {
  nccl_uct_comm_t *comm = reg_comm;
  uct_component_h comp  = comm->uct_iface->comp;
  uct_md_h md           = comm->uct_iface->md;
  intptr_t addr         = (intptr_t)data;
  size_t rkey_size      = comm->uct_iface->rkey_packed_size;
  nccl_uct_memh_t *uct_memh;

  NCCLCHECK(ncclIbMalloc((void**)&uct_memh, sizeof(*uct_memh) + rkey_size));
  uct_memh->comm = comm;

  /* Use integral pages */
  size += addr & (NCCL_UCT_REG_ALIGN - 1);
  size  = (size + NCCL_UCT_REG_ALIGN - 1) & ~(NCCL_UCT_REG_ALIGN - 1);
  addr &= ~(NCCL_UCT_REG_ALIGN - 1);

  /* Register memory */
  UCXCHECK(uct_md_mem_reg(md, (void*)addr, size, UCT_MD_MEM_ACCESS_ALL,
                          &uct_memh->memh),
           return ncclSystemError, "register %p/%zu on comm %p", addr, size,
           comm);
  /* Pack memory */
  UCXCHECK(uct_md_mkey_pack(md, uct_memh->memh, uct_memh->rkey),
           return ncclSystemError, "pack rkey for %p/%zu on comm %p", addr,
           size, comm);
  /* Unpack rkey from sender side */
  UCXCHECK(uct_rkey_unpack(comp, uct_memh->rkey, &uct_memh->bundle),
           return ncclInternalError, "unpack rkey");

  *mhandle = uct_memh;
  return ncclSuccess;
}

ncclResult_t nccl_uct_reg_mr_dmabuf(void *reg_comm, void *data, size_t size,
                                    int type, uint64_t offset, int fd,
                                    void **mhandle) {
  return nccl_uct_reg_mr(reg_comm, data, size, type, mhandle);
}

ncclResult_t nccl_uct_dereg_mr(void *dereg_comm, void *mhandle) {
  nccl_uct_comm_t *comm     = dereg_comm;
  uct_component_h comp      = comm->uct_iface->comp;
  nccl_uct_memh_t *uct_memh = mhandle;

  assert(uct_memh->memh != UCT_MEM_HANDLE_NULL);
  assert(uct_memh->comm == comm);

  UCXCHECK(uct_rkey_release(comp, &uct_memh->bundle), , "release rkey bundle");
  UCXCHECK(uct_md_mem_dereg(comm->uct_iface->md, uct_memh->memh),
           return ncclSystemError, "deregister memh %p on comm %p", uct_memh,
           comm);

  uct_memh->comm = NULL;
  free(uct_memh);
  return ncclSuccess;
}

int nccl_uct_flush_index(nccl_uct_comm_t *base, int *sizes, int n) {
  int last = -1;
  int i;

  if (base->gpu_flush.enabled) {
    for (i = 0; i < n; i++) {
      if (sizes[i]) {
        last = i;
      }
    }
  }

  return last;
}

ncclResult_t nccl_uct_flush(nccl_uct_comm_t *base_comm, void *data, int size,
                            nccl_uct_memh_t *uct_memh,
                            uct_completion_t *completion, void **request) {
  ucs_status_t status;
  uct_iov_t iov;

  iov.buffer = base_comm->gpu_flush.mem;
  iov.length = base_comm->uct_iface->min_get_zcopy? : 1;
  iov.memh   = base_comm->gpu_flush.memh;
  iov.stride = 0;
  iov.count  = 1;

  assert(size >= iov.length);

  status = uct_ep_get_zcopy(base_comm->gpu_flush.uct_ep->ep, &iov, 1,
                            (uint64_t)data, uct_memh->bundle.rkey, completion);
  if (status != UCS_INPROGRESS) {
    *request = NULL;
    if (status != UCS_OK) {
      WARN("Failed to flush local ep comm=%p status=%d", base_comm, status);
      return ncclInternalError;
    }
  }

  return ncclSuccess;
}

static void nccl_uct_worker_destroy(nccl_uct_worker_t *w) {
  nccl_uct_iface_close(w->uct_iface);
  uct_worker_destroy(w->worker);
  ucs_async_context_destroy(w->async);
  free(w);
}

static void nccl_uct_worker_put(nccl_uct_worker_t *worker) {
  nccl_uct_worker_t **wp;

  pthread_mutex_lock(&nccl_uct_lock);
  if (--worker->count < 1) {
    assert(worker->count == 0);
    for (wp = &worker->context->worker_list; *wp != NULL; wp = &(*wp)->next) {
      if (*wp == worker) {
        *wp = worker->next;
        nccl_uct_worker_destroy(worker);
        break;
      }
    }
  }
  pthread_mutex_unlock(&nccl_uct_lock);
}

void nccl_uct_comm_deinit(nccl_uct_comm_t *comm) {
  nccl_uct_ep_destroy(comm->uct_ep);

  if (comm->gpu_flush.uct_ep != NULL) {
    nccl_uct_ep_destroy(comm->gpu_flush.uct_ep);
    (void)uct_md_mem_dereg(comm->uct_iface->md, comm->gpu_flush.memh);
    free(comm->gpu_flush.mem);
  }
  nccl_uct_worker_put(comm->uct_worker);
}

ncclResult_t nccl_uct_get_properties(int dev, ncclNetProperties_t *props) {
  return nccl_p2p_ib_get_properties(ncclIbDevs, context.merge_dev_count, dev, props);
}

ncclResult_t nccl_uct_get_properties_v8(int dev, ncclNetProperties_v8_t *props_v8) {
  ncclNetProperties_t props;
  ncclResult_t ret = nccl_uct_get_properties(dev, &props);
  if (ret != ncclSuccess) {
    return ret;
  }

  props_v8->name             = props.name;
  props_v8->pciPath          = props.pciPath;
  props_v8->guid             = props.guid;
  props_v8->ptrSupport       = props.ptrSupport;
  props_v8->regIsGlobal      = props.regIsGlobal;
  props_v8->speed            = props.speed;
  props_v8->latency          = props.latency;
  props_v8->port             = props.port;
  props_v8->maxComms         = props.maxComms;
  props_v8->maxRecvs         = props.maxRecvs;
  props_v8->netDeviceType    = props.netDeviceType;
  props_v8->netDeviceVersion = props.netDeviceVersion;
  return ncclSuccess;
}

ncclResult_t nccl_uct_get_properties_v7(int dev,
                                        ncclNetProperties_v7_t *props_v7) {
  ncclNetProperties_t props;
  ncclResult_t ret = nccl_uct_get_properties(dev, &props);
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

ncclResult_t nccl_uct_reg_mr_v7(void *comm, void *data, int size, int type,
                                void **mhandle) {
  return nccl_uct_reg_mr(comm, data, (size_t)size, type, mhandle);
}

ncclResult_t nccl_uct_get_properties_v6(int dev,
                                        ncclNetProperties_v6_t *props_v6) {
  ncclNetProperties_t props;
  ncclResult_t ret = nccl_uct_get_properties(dev, &props);
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

ncclResult_t nccl_uct_connect_v6(int dev, void *handle, void **send_comm) {
  ncclNetDeviceHandle_t *dev_handle = NULL;
  return nccl_uct_connect(dev, handle, send_comm, &dev_handle);
}

ncclResult_t nccl_uct_accept_v6(void *listen_comm, void **recv_comm) {
  ncclNetDeviceHandle_v7_t *dev_handle = NULL;
  return nccl_uct_accept(listen_comm, recv_comm, &dev_handle);
}
