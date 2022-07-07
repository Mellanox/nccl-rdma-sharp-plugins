/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdint.h>
#include "ibvwrap.h"
#include "nccl.h"

#define IBV_PTR_CHECK_ERRNO(call, retval, error_retval, name) \
  retval = call; \
  if (retval == error_retval) { \
    WARN("Call to " name " failed with error %s", strerror(errno)); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_PTR_CHECK(call, retval, error_retval, name) \
  retval = call; \
  if (retval == error_retval) { \
    WARN("Call to " name " failed"); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_INT_CHECK_RET_ERRNO(call, success_retval, name) \
  int ret = call; \
  if (ret != success_retval) { \
    WARN("Call to " name " failed with error %s", strerror(ret)); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_INT_CHECK(call, error_retval, name) \
  int ret = call; \
  if (ret == error_retval) { \
    WARN("Call to " name " failed"); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_PASSTHRU(call) \
  call; \
  return ncclSuccess;

ncclResult_t wrap_ibv_fork_init() {
  IBV_INT_CHECK(ibv_fork_init(), -1, "ibv_fork_init");
}

ncclResult_t wrap_ibv_get_device_list(struct ibv_device ***ret, int *num_devices) {
  *ret = ibv_get_device_list(num_devices);
  if (*ret == NULL) *num_devices = 0;
  return ncclSuccess;
}

ncclResult_t wrap_ibv_free_device_list(struct ibv_device **list) {
  IBV_PASSTHRU(ibv_free_device_list(list));
}

const char *wrap_ibv_get_device_name(struct ibv_device *device) {
  return ibv_get_device_name(device);
}

ncclResult_t wrap_ibv_open_device(struct ibv_context **ret, struct ibv_device *device) { /*returns 0 on success, -1 on failure*/
  IBV_PTR_CHECK(ibv_open_device(device), *ret, NULL, "ibv_open_device");
}

ncclResult_t wrap_ibv_close_device(struct ibv_context *context) { /*returns 0 on success, -1 on failure*/
  IBV_INT_CHECK(ibv_close_device(context), -1, "ibv_close_device");
}

ncclResult_t wrap_ibv_get_async_event(struct ibv_context *context, struct ibv_async_event *event) { /*returns 0 on success, and -1 on error*/
  IBV_INT_CHECK(ibv_get_async_event(context, event), -1, "ibv_get_async_event");
}

ncclResult_t wrap_ibv_ack_async_event(struct ibv_async_event *event) {
  IBV_PASSTHRU(ibv_ack_async_event(event));
}

ncclResult_t wrap_ibv_query_device(struct ibv_context *context, struct ibv_device_attr *device_attr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibv_query_device(context, device_attr), 0, "ibv_query_device");
}

ncclResult_t wrap_ibv_query_port(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibv_query_port(context, port_num, port_attr), 0, "ibv_query_port");
}

ncclResult_t wrap_ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid) {
  IBV_INT_CHECK_RET_ERRNO(ibv_query_gid(context, port_num, index, gid), 0, "ibv_query_gid");
}

ncclResult_t wrap_ibv_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask, struct ibv_qp_init_attr *init_attr) {
  IBV_INT_CHECK_RET_ERRNO(ibv_query_qp(qp, attr, attr_mask, init_attr), 0, "ibv_query_qp");
}

ncclResult_t wrap_ibv_alloc_pd(struct ibv_pd **ret, struct ibv_context *context) {
  IBV_PTR_CHECK(ibv_alloc_pd(context), *ret, NULL, "ibv_alloc_pd");
}

ncclResult_t wrap_ibv_dealloc_pd(struct ibv_pd *pd) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibv_dealloc_pd(pd), 0, "ibv_dealloc_pd");
}

ncclResult_t wrap_ibv_reg_mr(struct ibv_mr **ret, struct ibv_pd *pd, void *addr, size_t length, int access) {
  IBV_PTR_CHECK(ibv_reg_mr(pd, addr, length, access), *ret, NULL, "ibv_reg_mr");
}

struct ibv_mr * wrap_direct_ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access) {
  return ibv_reg_mr(pd, addr, length, access);
}

ncclResult_t wrap_ibv_reg_mr_iova2(struct ibv_mr **ret, struct ibv_pd *pd, void *addr, size_t length, uint64_t iova, int access) {
#if HAVE_DECL_IBV_ACCESS_RELAXED_ORDERING
  IBV_PTR_CHECK(ibv_reg_mr_iova2(pd, addr, length, iova, access), *ret, NULL, "ibv_reg_mr_iova2");
#else
  return ncclSystemError;
#endif
}

ncclResult_t wrap_ibv_dereg_mr(struct ibv_mr *mr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibv_dereg_mr(mr), 0, "ibv_dereg_mr");
}

ncclResult_t wrap_ibv_create_cq(struct ibv_cq **ret, struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel, int comp_vector) {
  IBV_PTR_CHECK(ibv_create_cq(context, cqe, cq_context, channel, comp_vector), *ret, NULL, "ibv_create_cq");
}

ncclResult_t wrap_ibv_destroy_cq(struct ibv_cq *cq) {
  IBV_INT_CHECK_RET_ERRNO(ibv_destroy_cq(cq), 0, "ibv_destroy_cq");
}

ncclResult_t wrap_ibv_destroy_qp(struct ibv_qp *qp) {
  IBV_INT_CHECK_RET_ERRNO(ibv_destroy_qp(qp), 0, "ibv_destroy_qp");
}

ncclResult_t wrap_ibv_create_qp(struct ibv_qp **ret, struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr) {
  IBV_PTR_CHECK(ibv_create_qp(pd, qp_init_attr), *ret, NULL, "ibv_create_qp");
}

ncclResult_t wrap_ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibv_modify_qp(qp, attr, attr_mask), 0, "ibv_modify_qp");
}

ncclResult_t wrap_ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad_wr) {
  IBV_INT_CHECK_RET_ERRNO(qp->context->ops.post_send(qp, wr, bad_wr), 0, "ibv_post_send");
}

ncclResult_t wrap_ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad_wr) {
  IBV_INT_CHECK_RET_ERRNO(qp->context->ops.post_recv(qp, wr, bad_wr), 0, "ibv_post_recv");
  return ncclSuccess;
}

ncclResult_t wrap_ibv_event_type_str(char **ret, enum ibv_event_type event) {
  *ret = (char *) ibv_event_type_str(event);
  return ncclSuccess;
}
