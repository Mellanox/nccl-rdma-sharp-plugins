/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_UCX_UCT_LIB_H_
#define NCCL_UCX_UCT_LIB_H_

#include <assert.h>
#include <stdint.h>
#include <unistd.h>

#include "p2p_plugin.h"
#include "socket.h"

#include <uct/api/uct.h>

#define NCCL_UCX_UCT_MAX_RECVS       NCCL_NET_IB_MAX_RECVS
#define NCCL_UCT_LISTEN_HANDLE_MAGIC 0x43cf19ed91abdb85
#define NCCL_UCT_REG_ALIGN           4096

typedef enum {
  NCCL_UCT_AM_RTR = 14, /* Use particular values */
  NCCL_UCT_AM_ATP = 15,
  NCCL_UCT_AM_RTS = 16,
  NCCL_UCT_AM_ATS = 17
} nccl_uct_am_type_t;

typedef enum {
  NCCL_UCT_START = 0,
  NCCL_UCT_CONNECT,
  NCCL_UCT_ACCEPT,
  NCCL_UCT_RECEIVE_REMOTE, /* Acceptor receives ep addr/remote communicator */
  NCCL_UCT_RECEIVE_ADDR,
  NCCL_UCT_RX_READY,
  NCCL_UCT_DONE
} nccl_uct_state_t;

/* UCT EP address to exchange and connect to */
typedef struct {
  uint8_t dev_addr_size;
  uint8_t ep_addr_size;
  uint8_t data[64];
} nccl_uct_ep_addr_t;

typedef struct {
  uct_iface_h     iface;
  uct_md_h        md;
  uct_component_h comp;
  void            *addr;
  size_t          addr_size;
  void            *dev_addr;
  size_t          dev_addr_size;
  size_t          ep_addr_size;
  size_t          rkey_packed_size;

  size_t          am_max_short;
  size_t          min_get_zcopy;
} nccl_uct_iface_t;

struct nccl_uct_context;

typedef struct nccl_uct_worker {
  struct nccl_uct_worker *next;
  struct {
    pthread_t thread;
    int       dev;
  } id;

  int                     count;
  ucs_async_context_t     *async;
  uct_worker_h            worker;
  nccl_uct_iface_t        *uct_iface;
  struct nccl_uct_context *context;
} nccl_uct_worker_t;

typedef struct {
  uct_ep_h         ep;
  uct_ep_addr_t    *addr;
  size_t           addr_size;
  nccl_uct_iface_t *uct_iface;
  uint8_t          data[];
} nccl_uct_ep_t;

/* All the remote addresses for the communicator */
typedef struct nccl_uct_comm_addr {
  nccl_uct_ep_addr_t rma;
  /* TODO: Add multi-QP here */
} nccl_uct_comm_addr_t;

/* Either Receiver or Sender communicator, connected to one peer */
typedef struct nccl_uct_comm {
  struct ncclSocket       sock;
  struct nccl_uct_context *context;
  int                     dev;

  nccl_uct_worker_t       *uct_worker;
  nccl_uct_iface_t        *uct_iface;
  nccl_uct_ep_t           *uct_ep;

  struct nccl_uct_comm_remote {
    nccl_uct_comm_addr_t       addr;  /* Remote addresses */
    const struct nccl_uct_comm *comm; /* Cookie received in connect */
  } remote;

  /* Local GET on current device */
  struct {
    int                enabled;
    nccl_uct_ep_t      *uct_ep; /* Locally read from HCA */
    nccl_uct_ep_addr_t addr;

    uint8_t            *mem; /* Dummy memory to read into */
    uct_mem_h          memh;
  } gpu_flush;
} nccl_uct_comm_t;

/* State tracking used while connecting/accepting only */
typedef struct {
  nccl_uct_state_t state;
  nccl_uct_comm_t  *comm;  /* current communicator being created */
  int              offset; /* for Socket reading */
  int              ready;  /* accept must complete after connect */
} nccl_uct_stage_t;

/* Memory registration handle in NCCL UCT plugin returned by ->regMR() */
typedef struct {
  uct_mem_h         memh;
  nccl_uct_comm_t   *comm;
  uct_rkey_bundle_t bundle;
  uint8_t           rkey[];
} nccl_uct_memh_t;

/* On-the-wire handle passed OOB by NCCL from listener to connector */
typedef struct {
  uint64_t                  magic;
  struct {
    union ncclSocketAddress addr;
    uint32_t                id;
  } listener;
  nccl_uct_comm_t           *comm; /* Created communicator in accept */
  nccl_uct_stage_t          stage; /* Used by connector */
} nccl_uct_listen_handle_t;

/* Communicator while listening to remote ranks */
typedef struct {
  struct ncclSocket       sock;
  struct nccl_uct_context *context;
  int                     dev;
  uint32_t                id;
  nccl_uct_worker_t       *uct_worker;
  nccl_uct_comm_t         *comm;

  /* Used by acceptor */
  nccl_uct_stage_t        stage;
} nccl_uct_listen_comm_t;

/* Global state of the plugin */
typedef struct nccl_uct_context {
  /* Transport to use */
  const char              *tl_name;

  /* IB devices available */
  int                     dev_count;
  int                     merge_dev_count;

  /* Use by common code to setup communicators */
  struct nccl_uct_ops {
    ncclResult_t (*comm_alloc)(nccl_uct_comm_t **comm);
    ncclResult_t (*comm_init)(nccl_uct_comm_t *comm,
                              struct nccl_uct_context *context,
                              nccl_uct_worker_t *worker, int dev,
                              const nccl_uct_comm_t *remote_comm);
    ncclResult_t (*iface_set)(nccl_uct_iface_t *uct_iface);
  } ops;

  /* Max sizes needed */
  size_t                  am_short_size;
  size_t                  rkey_size;

  /* OOB socket for accepting/connecting */
  char                    if_name[MAX_IF_NAME_SIZE];
  union ncclSocketAddress if_addr;

  /* Number of listener created */
  uint32_t                listener_count;

  /* List of created workers */
  nccl_uct_worker_t       *worker_list;
} nccl_uct_context_t;

#define UCXCHECK(statement, failure_action, message, ...) \
  do { \
    ucs_status_t _status = statement; \
    if (_status != UCS_OK) { \
      WARN("Failed: " message ": %s", ##__VA_ARGS__, \
           ucs_status_string(_status)); \
      failure_action; \
    } \
  } while (0)

extern nccl_uct_context_t context;

/* Library functions */
ncclResult_t nccl_uct_iface_set_handler(nccl_uct_iface_t *uct_iface, int id,
                                        uct_am_callback_t callback);
ncclResult_t nccl_uct_devices(int *ndev);
ncclResult_t nccl_uct_comm_init(nccl_uct_comm_t *comm,
                                nccl_uct_context_t *context,
                                nccl_uct_worker_t *worker, int dev,
                                const nccl_uct_comm_t *remote_comm);
void nccl_uct_comm_deinit(nccl_uct_comm_t *comm);
int nccl_uct_flush_index(nccl_uct_comm_t *base, int *sizes, int n);
ncclResult_t nccl_uct_flush(nccl_uct_comm_t *base_comm, void *data, int size,
                            nccl_uct_memh_t *uct_memh,
                            uct_completion_t *completion, void **request);
void nccl_uct_empty_callback(uct_completion_t *comp);

/* NCCL common plugin callbacks */
ncclResult_t nccl_uct_listen(int dev, void *listen_handle, void **listen_comm);
ncclResult_t nccl_uct_accept(void *listen_comm, void **recv_comm,
                             ncclNetDeviceHandle_v7_t **recvDevComm);
ncclResult_t nccl_uct_connect(int dev, void *listen_handle, void **send_comm,
                              ncclNetDeviceHandle_t **sendDevComm);
ncclResult_t nccl_uct_close_listen(void *listen_comm);
ncclResult_t nccl_uct_reg_mr_dmabuf(void *reg_comm, void *data, size_t size,
                                    int type, uint64_t offset, int fd,
                                    void **mhandle);
ncclResult_t nccl_uct_reg_mr(void *reg_comm, void *data, size_t size, int type,
                             void **mhandle);
ncclResult_t nccl_uct_dereg_mr(void *dereg_comm, void *mhandle);

/* Compatibility callback */
ncclResult_t nccl_uct_get_properties_v8(int dev,
                                        ncclNetProperties_v8_t *props_v8);
ncclResult_t nccl_uct_get_properties_v7(int dev,
                                        ncclNetProperties_v7_t *props_v7);
ncclResult_t nccl_uct_reg_mr_v7(void *comm, void *data, int size, int type,
                                void **mhandle);
ncclResult_t nccl_uct_get_properties_v6(int dev,
                                        ncclNetProperties_v6_t *props_v6);
ncclResult_t nccl_uct_connect_v6(int dev, void *handle, void **send_comm);
ncclResult_t nccl_uct_accept_v6(void *listen_comm, void **recv_comm);
ncclResult_t nccl_uct_get_properties(int dev, ncclNetProperties_t *props);


#define NCCL_UCT_PLUGIN_BASE(plugin_name, prefix, get_properties_func, \
                             connect_func, accept_func, reg_mr_func, \
                             isend_func, irecv_func) \
  { \
    .name          = plugin_name, \
    .init          = prefix##_init, \
    .devices       = nccl_uct_devices, \
    .getProperties = get_properties_func, \
    .listen        = nccl_uct_listen, \
    .connect       = connect_func, \
    .accept        = accept_func, \
    .regMr         = reg_mr_func, \
    .regMrDmaBuf   = nccl_uct_reg_mr_dmabuf, \
    .deregMr       = nccl_uct_dereg_mr, \
    .isend         = prefix##_##isend_func, \
    .irecv         = prefix##_##irecv_func, \
    .iflush        = prefix##_iflush, \
    .test          = prefix##_test, \
    .closeSend     = prefix##_close, \
    .closeRecv     = prefix##_close, \
    .closeListen   = nccl_uct_close_listen \
  }

#define NCCL_UCT_PLUGIN_V9(plugin_name, prefix) \
  NCCL_UCT_PLUGIN_BASE(plugin_name, prefix, nccl_uct_get_properties, \
                       nccl_uct_connect, nccl_uct_accept, nccl_uct_reg_mr, \
                       isend, irecv)

#define NCCL_UCT_PLUGIN_V8(plugin_name, prefix) \
  NCCL_UCT_PLUGIN_BASE(plugin_name, prefix, nccl_uct_get_properties_v8, \
                       nccl_uct_connect, nccl_uct_accept, nccl_uct_reg_mr, \
                       isend_v8, irecv_v8)

#define NCCL_UCT_PLUGIN_V7(plugin_name, prefix) \
  NCCL_UCT_PLUGIN_BASE(plugin_name, prefix, nccl_uct_get_properties_v7, \
                       nccl_uct_connect, nccl_uct_accept, nccl_uct_reg_mr_v7, \
                       isend_v8, irecv_v8)

#define NCCL_UCT_PLUGIN_V6(plugin_name, prefix) \
  NCCL_UCT_PLUGIN_BASE(plugin_name, prefix, nccl_uct_get_properties_v6, \
                       nccl_uct_connect_v6, nccl_uct_accept_v6, \
                       nccl_uct_reg_mr_v7, isend_v8, irecv_v8)

#define NCCL_UCT_PLUGIN_V5(plugin_name, prefix) \
  { \
    .name          = plugin_name, \
    .init          = prefix##_init, \
    .devices       = nccl_uct_devices, \
    .getProperties = nccl_uct_get_properties_v6, \
    .listen        = nccl_uct_listen, \
    .connect       = nccl_uct_connect_v6, \
    .accept        = nccl_uct_accept_v6, \
    .regMr         = nccl_uct_reg_mr_v7, \
    .deregMr       = nccl_uct_dereg_mr, \
    .isend         = prefix##_isend_v8, \
    .irecv         = prefix##_irecv_v8, \
    .iflush        = prefix##_iflush, \
    .test          = prefix##_test, \
    .closeSend     = prefix##_close, \
    .closeRecv     = prefix##_close, \
    .closeListen   = nccl_uct_close_listen \
  }

#endif /* NCCL_UCX_UCT_LIB_H_ */
