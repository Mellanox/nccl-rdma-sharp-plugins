/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 * Copyright (C) 2019-2020, Mellanox Technologies Ltd. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdint.h>

#include "nccl.h"
#include "nccl_net.h"
#include "debug.h"

extern ncclDebugLogger_t pluginLogFunction;

ncclResult_t nccl_ucx_init(ncclDebugLogger_t logFunction) {
  WARN("NET/UCX: not implemented");
  return ncclInternalError;
}

ncclResult_t nccl_ucx_devices(int* ndev) {
  WARN("NET/UCX: not implemented");
  return ncclInternalError;
}

ncclResult_t nccl_ucx_get_properties(int dev, ncclNetProperties_t* props) {
  WARN("NET/UCX: not implemented");
  return ncclInternalError;
}

ncclResult_t nccl_ucx_listen(int dev, void *handle, void **listen_comm) {
  WARN("NET/UCX: not implemented");
  return ncclInternalError;
}

ncclResult_t nccl_ucx_connect(int dev, void *handle, void **send_comm) {
  WARN("NET/UCX: not implemented");
  return ncclInternalError;
}

ncclResult_t nccl_ucx_accept(void *listen_comm, void **recv_comm) {
  WARN("NET/UCX: not implemented");
  return ncclInternalError;
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

ncclResult_t nccl_ucx_close_send(void *send_comm) {
  WARN("NET/UCX: not implemented");
  return ncclInternalError;
}

ncclResult_t nccl_ucx_close_recv(void *recv_comm) {
  WARN("NET/UCX: not implemented");
  return ncclInternalError;
}

ncclResult_t nccl_ucx_close_listen(void *listen_comm) {
  WARN("NET/UCX: not implemented");
  return ncclInternalError;
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
