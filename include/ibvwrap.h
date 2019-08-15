/*************************************************************************
 * Copyright (c) 2004, 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2004, 2011-2012 Intel Corporation.  All rights reserved.
 * Copyright (c) 2005, 2006, 2007 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2005 PathScale, Inc.  All rights reserved.
 *
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_IBVWRAP_H_
#define NCCL_IBVWRAP_H_

#include "core.h"
#include <infiniband/verbs.h>

ncclResult_t wrap_ibv_fork_init(void);
ncclResult_t wrap_ibv_get_device_list(struct ibv_device ***ret, int *num_devices);
ncclResult_t wrap_ibv_free_device_list(struct ibv_device **list);
const char *wrap_ibv_get_device_name(struct ibv_device *device);
ncclResult_t wrap_ibv_open_device(struct ibv_context **ret, struct ibv_device *device);
ncclResult_t wrap_ibv_close_device(struct ibv_context *context);
ncclResult_t wrap_ibv_get_async_event(struct ibv_context *context, struct ibv_async_event *event);
ncclResult_t wrap_ibv_ack_async_event(struct ibv_async_event *event);
ncclResult_t wrap_ibv_query_device(struct ibv_context *context, struct ibv_device_attr *device_attr);
ncclResult_t wrap_ibv_query_port(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr);
ncclResult_t wrap_ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid);
ncclResult_t wrap_ibv_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask, struct ibv_qp_init_attr *init_attr);
ncclResult_t wrap_ibv_alloc_pd(struct ibv_pd **ret, struct ibv_context *context);
ncclResult_t wrap_ibv_dealloc_pd(struct ibv_pd *pd);
ncclResult_t wrap_ibv_reg_mr(struct ibv_mr **ret, struct ibv_pd *pd, void *addr, size_t length, int access);
struct ibv_mr * wrap_direct_ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access);
ncclResult_t wrap_ibv_dereg_mr(struct ibv_mr *mr);
ncclResult_t wrap_ibv_create_comp_channel(struct ibv_comp_channel **ret, struct ibv_context *context);
ncclResult_t wrap_ibv_destroy_comp_channel(struct ibv_comp_channel *channel);
ncclResult_t wrap_ibv_create_cq(struct ibv_cq **ret, struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel, int comp_vector);
ncclResult_t wrap_ibv_destroy_cq(struct ibv_cq *cq);
static inline ncclResult_t wrap_ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc, int* num_done) {
  int done = cq->context->ops.poll_cq(cq, num_entries, wc); /*returns the number of wcs or 0 on success, a negative number otherwise*/
  if (done < 0) {
    WARN("Call to ibv_poll_cq() returned %d", done);
    return ncclSystemError;
  }
  *num_done = done;
  return ncclSuccess;
}
ncclResult_t wrap_ibv_create_qp(struct ibv_qp **ret, struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr);
ncclResult_t wrap_ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask);
ncclResult_t wrap_ibv_destroy_qp(struct ibv_qp *qp);
ncclResult_t wrap_ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr);
ncclResult_t wrap_ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad_wr);
ncclResult_t wrap_ibv_event_type_str(char **ret, enum ibv_event_type event);

#endif //End include guard
