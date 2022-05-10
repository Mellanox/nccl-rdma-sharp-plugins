/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>

#include "config.h"
#include "core.h"
#include "nccl.h"
#include "nccl_net.h"
#include "p2p_plugin.h"
#include "param.h"
#include "sharp/api/version.h"
#include "sharp/api/sharp_coll.h"
#include "utils.h"

extern ncclNet_t NCCL_PLUGIN_SYMBOL;
int ncclNSharpDevs = -1;
struct sharp_coll_caps sharp_caps;
NCCL_PARAM(SharpGroupSizeThresh, "SHARP_GROUP_SIZE_THRESH", 2);
NCCL_PARAM(SharpV3Datatypes, "SHARP_V3_DATATYPES", 1);

enum ncclSharpRequestType {
  NCCL_SHARP_REQ_SHARP_COLL,
  NCCL_SHARP_REQ_IFLUSH,
};

struct ncclSharpRequest {
  int requestType;
  void *sharpRequest;
  int  size;
  int  used;
};

struct ncclSharpListenComm {
  int   dev;
  void *listenCommP2P;
};

struct ncclSharpCollComm {
  int    rank;
  int    nranks;
  void*  recvComm;
  void*  sendComm;
  struct ncclSharpRequest*   reqs;
  struct sharp_coll_context* sharpCollContext;
  struct sharp_coll_comm*    sharpCollComm;
};

struct ncclSharpMemHandle{
  void *mr;
  void *ncclIbMr;
  int  type;
};

struct ncclSharpInfo {
  uint64_t hostId;
  uint64_t jobId;
};

static __inline__ enum sharp_datatype typeConvert(ncclDataType_t type) {
  switch (type) {
    case ncclFloat16: return SHARP_DTYPE_FLOAT_SHORT;
    case ncclInt32: return SHARP_DTYPE_INT;
    case ncclUint32: return SHARP_DTYPE_UNSIGNED;
    case ncclFloat32: return SHARP_DTYPE_FLOAT;
    case ncclInt64: return SHARP_DTYPE_LONG;
    case ncclUint64: return SHARP_DTYPE_UNSIGNED_LONG;
    case ncclFloat64: return SHARP_DTYPE_DOUBLE;
#ifdef HAVE_SHARP_DTYPE_BFLOAT16_UINT8_INT8
    case ncclBfloat16: return (ncclParamSharpV3Datatypes() ? SHARP_DTYPE_BFLOAT16 : SHARP_DTYPE_NULL);
    case ncclInt8: return (ncclParamSharpV3Datatypes() ? SHARP_DTYPE_INT8 : SHARP_DTYPE_NULL);
    case ncclUint8: return (ncclParamSharpV3Datatypes() ? SHARP_DTYPE_UINT8 : SHARP_DTYPE_NULL);
#endif
    default: return SHARP_DTYPE_NULL;
  }
}

static __inline__ int typeSize(ncclDataType_t type) {
  switch (type) {
    case ncclFloat16: return 2;
    case ncclInt32: return 4;
    case ncclUint32: return 4;
    case ncclFloat32: return 4;
    case ncclInt64: return 8;
    case ncclUint64: return 8;
    case ncclFloat64: return 8;
    case ncclBfloat16: return 2;
    case ncclInt8: return 1;
    case ncclUint8: return 1;
    default:
      WARN("SHARP: unsupported data type\n");
      return -1;
  }
}

static __inline__ enum sharp_reduce_op opConvert(ncclRedOp_t op) {
  switch (op) {
    case ncclSum: return SHARP_OP_SUM;
    case ncclMax: return SHARP_OP_MAX;
    case ncclMin: return SHARP_OP_MIN;
    default: return SHARP_OP_NULL;
  }
}

int ncclSharpAllGather(void *context, void *buf, int len) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)context;
  nccl_p2p_plugin_t p2p_plugin;
  void* rMhandle = NULL, *sMhandle = NULL;

  assert(cComm->recvComm != NULL);
  assert(cComm->sendComm != NULL);

  p2p_plugin = nccl_p2p_get_plugin_type();
  if (p2p_plugin != NCCL_P2P_UCX) {
    NCCLCHECK(NCCL_PLUGIN_SYMBOL.regMr(cComm->recvComm, buf, cComm->nranks*len, NCCL_PTR_HOST, &rMhandle));
    NCCLCHECK(NCCL_PLUGIN_SYMBOL.regMr(cComm->sendComm, buf, cComm->nranks*len, NCCL_PTR_HOST, &sMhandle));
  }

  int speer = cComm->rank;
  for (int i=0; i<cComm->nranks-1; i++) {
    void* srequest = NULL, *rrequest = NULL;
    int rpeer = (speer-1+cComm->nranks)%cComm->nranks;
    while (srequest == NULL || rrequest == NULL) {
       void *rbuf = ((char*)buf)+rpeer*len;
       int tag = 0x69;
       if (srequest == NULL) NCCLCHECK(NCCL_PLUGIN_SYMBOL.isend(cComm->sendComm, ((char*)buf)+speer*len, len, tag, sMhandle, &srequest));
       if (rrequest == NULL) NCCLCHECK(NCCL_PLUGIN_SYMBOL.irecv(cComm->recvComm, 1, &rbuf, &len, &tag, &rMhandle, &rrequest));
    }
    while (srequest || rrequest) {
      int done;
      if (rrequest) NCCLCHECK(NCCL_PLUGIN_SYMBOL.test(rrequest, &done, NULL));
      if (done) rrequest = NULL;
      if (srequest) NCCLCHECK(NCCL_PLUGIN_SYMBOL.test(srequest, &done, NULL));
      if (done) srequest = NULL;
    }
    speer = rpeer;
  }
  if (p2p_plugin != NCCL_P2P_UCX) {
    NCCLCHECK(NCCL_PLUGIN_SYMBOL.deregMr(cComm->recvComm, rMhandle));
    NCCLCHECK(NCCL_PLUGIN_SYMBOL.deregMr(cComm->sendComm, sMhandle));
  }

  return 0;
}

int ncclSharpOobBarrier(void *ctx) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)ctx;
  int* dummy;
  NCCLCHECK(ncclIbMalloc((void**)&dummy, cComm->nranks*sizeof(int)));
  NCCLCHECK(ncclSharpAllGather(ctx, dummy, sizeof(int)));
  free(dummy);
  return 0;
}

int ncclSharpOobGather(void *ctx, int root, void *sbuf, void *rbuf, int size) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)ctx;
  int nranks = cComm->nranks;
  void *tmp;
  NCCLCHECK(ncclIbMalloc(&tmp, nranks*size));
  memcpy((void*)((ptrdiff_t)tmp + size*cComm->rank), sbuf, size);
  NCCLCHECK(ncclSharpAllGather(cComm, tmp, size));
  if (cComm->rank == root) {
    memcpy(rbuf, tmp, nranks*size);
  }
  free(tmp);
  return 0;
}

int ncclSharpOobBcast(void *ctx, void *buf, int size, int root) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)ctx;
  void *tmp;
  NCCLCHECK(ncclIbMalloc(&tmp, size*cComm->nranks));
  if (cComm->rank == root) {
    memcpy((void*)((ptrdiff_t)tmp+size*cComm->rank), buf, size);
  }
  NCCLCHECK(ncclSharpAllGather(cComm, tmp, size));
  if (cComm->rank != root) {
    memcpy(buf, (void*)((ptrdiff_t)tmp+size*root), size);
  }
  free(tmp);
  return 0;
}

ncclResult_t ncclSharpInit(ncclDebugLogger_t logFunction) {
  struct timeval tval;
  gettimeofday(&tval, NULL);
  srand((int) tval.tv_usec);

  /* set SHARP COLL library default for plugin */
  setenv("SHARP_COLL_ENABLE_SAT", "1", 0);
  setenv("SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD", "0", 0);
  setenv("SHARP_COLL_LOCK_ON_COMM_INIT", "1", 0);
  setenv("SHARP_COLL_LOG_LEVEL", "3", 0);

  return NCCL_PLUGIN_SYMBOL.init(logFunction);
}

ncclResult_t ncclSharpDevices(int* ndev) {
  *ndev = ncclNSharpDevs;
  return ncclSuccess;
}

ncclResult_t ncclSharpGetProperties(int dev, ncclNetProperties_t* props) {
  return NCCL_PLUGIN_SYMBOL.getProperties(dev, props);
}

ncclResult_t ncclSharpListen(int dev, void* opaqueHandle, void** listenComm) {
  struct ncclSharpListenComm *lComm;
  ncclResult_t status;

  NCCLCHECK(ncclIbMalloc((void**)&lComm, sizeof(struct ncclSharpListenComm)));
  status = NCCL_PLUGIN_SYMBOL.listen(dev, opaqueHandle, &lComm->listenCommP2P);
  lComm->dev = dev;
  *listenComm = lComm;
  return status;
}

ncclResult_t ncclSharpConnect(void* handles[], int nranks, int rank, void* listenComm, void** collComm) {
  struct ncclSharpListenComm* lComm = (struct ncclSharpListenComm*)listenComm;
  struct ncclSharpCollComm* cComm;
  char *useSharp;

  if(nranks < ncclParamSharpGroupSizeThresh()) {
    INFO(NCCL_INIT|NCCL_NET|NCCL_ENV, "SHARP: Group size:%d is less than threshold:%d. fallback to non-sharp",
         nranks, ncclParamSharpGroupSizeThresh());
    return ncclInvalidUsage;
  }

  useSharp = getenv("NCCL_SHARP_DISABLE");
  if(useSharp != NULL) {
    if(strcmp(useSharp, "1") == 0) {
      INFO(NCCL_INIT|NCCL_NET|NCCL_ENV, "SHARP: Set to disable on this communicator");
      return ncclInvalidUsage;
    }
  }

  NCCLCHECK(ncclIbMalloc((void**)&cComm, sizeof(struct ncclSharpCollComm)));
  NCCLCHECK(ncclIbMalloc((void**)&cComm->reqs, sizeof(struct ncclSharpRequest)*MAX_REQUESTS));

  cComm->nranks = nranks;
  cComm->rank = rank;
  if (cComm->rank == -1) {
    WARN("Could not determine my rank\n");
    return ncclInternalError;
  }
  int next = (cComm->rank + 1) % nranks;
  do {
    NCCLCHECK(NCCL_PLUGIN_SYMBOL.connect(lComm->dev, handles[next], &cComm->sendComm));
  } while(cComm->sendComm == NULL);

  do {
    NCCLCHECK(NCCL_PLUGIN_SYMBOL.accept(lComm->listenCommP2P, &cComm->recvComm)); // From prev
  } while(cComm->recvComm == NULL);

  struct ncclSharpInfo* allInfo;
  pid_t pid = getpid();
  pthread_t tid = pthread_self();
  NCCLCHECK(ncclIbMalloc((void**)&allInfo, sizeof(struct ncclSharpInfo)*nranks));
  allInfo[cComm->rank].hostId = gethostid();
  allInfo[cComm->rank].jobId = (((uint64_t)allInfo[cComm->rank].hostId << 32) | ((pid ^ tid) ^ rand()));
  NCCLCHECK(ncclSharpAllGather(cComm, allInfo, sizeof(struct ncclSharpInfo)));

  // Find my local rank;
  int localRank = 0;
  for (int i=0; i<cComm->rank; i++) {
    if (allInfo[cComm->rank].hostId == allInfo[i].hostId) {
      localRank++;
    }
  }
  uint64_t jobId = allInfo[0].jobId;
  free(allInfo);

  struct sharp_coll_init_spec init_spec = {0};
  init_spec.progress_func  = NULL;
  init_spec.job_id = jobId;
  init_spec.world_rank = cComm->rank;
  init_spec.world_size = nranks;
  init_spec.world_local_rank = 0;
  init_spec.enable_thread_support = 1;
  init_spec.group_channel_idx = 0;

  init_spec.oob_colls.barrier = ncclSharpOobBarrier;
  init_spec.oob_colls.bcast = ncclSharpOobBcast;
  init_spec.oob_colls.gather = ncclSharpOobGather;
  init_spec.oob_ctx = cComm;

  init_spec.config = sharp_coll_default_config;
  init_spec.config.user_progress_num_polls = 10000000;

  char devName[MAXNAMESIZE];
  ncclNetProperties_t prop;
  ncclSharpGetProperties(lComm->dev, &prop);
  snprintf(devName, MAXNAMESIZE, "%s:%d", prop.name, prop.port);
  init_spec.config.ib_dev_list = devName;

  int ret = sharp_coll_init(&init_spec, &cComm->sharpCollContext);

  INFO(NCCL_INIT, "SHARP rank %d/%d initialized on %s", cComm->rank, nranks, devName);

  if (ret < 0) {
    WARN("NET/IB : SHARP coll init error: %s(%d)\n", sharp_coll_strerror(ret), ret);
    return ncclInternalError;
  }

  struct sharp_coll_comm_init_spec comm_spec;
  comm_spec.rank = cComm->rank;
  comm_spec.size = nranks;
  comm_spec.oob_ctx = cComm;
  comm_spec.group_world_ranks = NULL;

  ret = sharp_coll_comm_init(cComm->sharpCollContext, &comm_spec, &cComm->sharpCollComm);
  if (ret < 0) {
    WARN("SHARP group create: %s(%d)\n", sharp_coll_strerror(ret), ret);
    sharp_coll_finalize(cComm->sharpCollContext);
    return ncclInternalError;
  }

  *collComm = cComm;
  return ncclSuccess;
}

ncclResult_t ncclSharpReduceSupport(ncclDataType_t dataType, ncclRedOp_t redOp, int* supported) {
  *supported = ((typeConvert(dataType) != SHARP_DTYPE_NULL) && (opConvert(redOp) != SHARP_OP_NULL));
  return ncclSuccess;
}

ncclResult_t ncclSharpRegMr(void* collComm, void* data, int size, int type, void** mhandle) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)collComm;

  struct ncclSharpMemHandle *mh;
  NCCLCHECK(ncclIbMalloc((void**)&mh, sizeof(struct ncclSharpMemHandle)));

  mh->type = type;
  if (SHARP_COLL_SUCCESS != sharp_coll_reg_mr(cComm->sharpCollContext, data, size, &(mh->mr))) {
    WARN("SHARP regmr failed\n");
    return ncclSystemError;
  }
  TRACE(NCCL_INIT,"sharpRegAddr %lx size %ld handle %x", data, size, mh->mr);

  NCCLCHECK(NCCL_PLUGIN_SYMBOL.regMr(cComm->recvComm, data, size, type, &mh->ncclIbMr));

  *mhandle = mh;
  return ncclSuccess;
}

ncclResult_t ncclSharpDeregMr(void* collComm, void* mhandle) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)collComm;
  struct ncclSharpMemHandle *mh = (struct ncclSharpMemHandle *)mhandle;

  if (SHARP_COLL_SUCCESS != sharp_coll_dereg_mr(cComm->sharpCollContext, mh->mr)) {
    WARN("SHARP deregmr failed\n");
  }

  NCCLCHECK(NCCL_PLUGIN_SYMBOL.deregMr(cComm->recvComm, mh->ncclIbMr));

  free(mh);
  return ncclSuccess;
}

ncclResult_t ncclSharpGetRequest(struct ncclSharpRequest* reqs, struct ncclSharpRequest** req) {
  for (int i=0; i<MAX_REQUESTS; i++) {
    struct ncclSharpRequest* r = reqs+i;
    if (r->used == 0) {
      r->used = 1;
      r->sharpRequest = NULL;
      r->size = 0;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("SHARP : unable to allocate request");
  *req = NULL;
  return ncclInternalError;
}

ncclResult_t ncclSharpIallreduce(void* collComm, void* sendData, void* recvData, int count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)collComm;

  enum sharp_datatype sharp_type = typeConvert(dataType);
  if (sharp_type == SHARP_DTYPE_NULL) {
    WARN("SHARP: unsupported data type\n");
    return ncclInternalError;
  }

  enum sharp_reduce_op op_type = opConvert(redOp);
  if (op_type == SHARP_OP_NULL) {
    WARN("SHARP: unsupported reduce operation\n");
    return ncclInternalError;
  }

  int dt_size = typeSize(dataType);
  struct ncclSharpMemHandle *mr_sbuf = (struct ncclSharpMemHandle*)sendMhandle;
  struct ncclSharpMemHandle *mr_rbuf = (struct ncclSharpMemHandle*)recvMhandle;

  struct ncclSharpRequest* req;
  NCCLCHECK(ncclSharpGetRequest(cComm->reqs, &req));

  struct sharp_coll_reduce_spec reduce_spec;

  reduce_spec.sbuf_desc.buffer.ptr = sendData;
  reduce_spec.sbuf_desc.buffer.length = count * dt_size;
  reduce_spec.sbuf_desc.buffer.mem_handle = mr_sbuf->mr;
  reduce_spec.sbuf_desc.type = SHARP_DATA_BUFFER;
  reduce_spec.sbuf_desc.mem_type = (mr_sbuf->type == NCCL_PTR_CUDA ? SHARP_MEM_TYPE_CUDA:SHARP_MEM_TYPE_HOST);

  reduce_spec.rbuf_desc.buffer.ptr = recvData;
  reduce_spec.rbuf_desc.buffer.length = count * dt_size;
  reduce_spec.rbuf_desc.buffer.mem_handle = mr_rbuf->mr;
  reduce_spec.rbuf_desc.type = SHARP_DATA_BUFFER;
  reduce_spec.rbuf_desc.mem_type = (mr_rbuf->type == NCCL_PTR_CUDA ? SHARP_MEM_TYPE_CUDA:SHARP_MEM_TYPE_HOST);

  reduce_spec.length = count;
  reduce_spec.dtype = sharp_type;
  reduce_spec.op = op_type;
  reduce_spec.aggr_mode = SHARP_AGGREGATION_NONE;

#if BLOCKING==0
  if (SHARP_COLL_SUCCESS != sharp_coll_do_allreduce_nb(cComm->sharpCollComm, &reduce_spec, &req->sharpRequest)) {
    WARN("SHARP allreduce failed\n");
  }
  req->size =  count * dt_size;
#else
  if (SHARP_COLL_SUCCESS != sharp_coll_do_allreduce(cComm->sharpCollComm, &reduce_spec)) {
    WARN("SHARP allreduce failed\n");
  }
  req->sharpRequest = (void *) 0xabababab;
  req->size =  count * dt_size;
#endif
  req->requestType = NCCL_SHARP_REQ_SHARP_COLL;
  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclSharpIflush(void* collComm, void* data, int size, void* mhandle, void **request) {
  struct ncclSharpCollComm *cComm = (struct ncclSharpCollComm*)collComm;
  struct ncclSharpMemHandle *mh = (struct ncclSharpMemHandle *)mhandle;
  struct ncclSharpRequest* req;

  NCCLCHECK(ncclSharpGetRequest(cComm->reqs, &req));
  req->requestType = NCCL_SHARP_REQ_IFLUSH;
  NCCL_PLUGIN_SYMBOL.iflush(cComm->recvComm, 1, &data, &size, &mh->ncclIbMr, &req->sharpRequest);
  if (!req->sharpRequest) {
    *request = NULL;
     req->used = 0;
     return ncclSuccess;
   }

  *request = req;
   return ncclSuccess;
}

ncclResult_t ncclSharpTest(void* request, int* done, int* size) {
  struct ncclSharpRequest* req = (struct ncclSharpRequest*)request;

  if (req->requestType == NCCL_SHARP_REQ_IFLUSH) {
    NCCL_PLUGIN_SYMBOL.test(req->sharpRequest, done, size);
    if (*done == 1) {
      req->used = 0;
    }
    return ncclSuccess;
  }

#if BLOCKING==0
  *done = sharp_coll_req_test(req->sharpRequest);
  if (*done){
    sharp_coll_req_free(req->sharpRequest);
    *size = req->size;
    req->used = 0;
  } else {
    *done = 0;
  }
#else
  if (req->size != -1) {
    *done = 1;
    *size = req->size;
    req->used = 0;
  } else {
     *done = 0;
  }
#endif
  return ncclSuccess;
}

ncclResult_t ncclSharpCloseColl(void* collComm) {
  struct ncclSharpCollComm* cComm = (struct ncclSharpCollComm*)collComm;

  sharp_coll_comm_destroy(cComm->sharpCollComm);
  sharp_coll_finalize(cComm->sharpCollContext);

  NCCLCHECK(NCCL_PLUGIN_SYMBOL.closeRecv(cComm->recvComm));
  NCCLCHECK(NCCL_PLUGIN_SYMBOL.closeSend(cComm->sendComm));
  free(cComm);
  return ncclSuccess;
}

ncclResult_t ncclSharpCloseListen(void* listenComm) {
  struct ncclSharpListenComm *lComm = (struct ncclSharpListenComm*)listenComm;
  ncclResult_t status;

  status = NCCL_PLUGIN_SYMBOL.closeListen(lComm->listenCommP2P);
  free(listenComm);
  return status;
}

ncclCollNet_t NCCL_COLLNET_PLUGIN_SYMBOL = {
  "SHARP",
  ncclSharpInit,
  ncclSharpDevices,
  ncclSharpGetProperties,
  ncclSharpListen,
  ncclSharpConnect,
  ncclSharpReduceSupport,
  ncclSharpRegMr,
  ncclSharpDeregMr,
  ncclSharpIallreduce,
  ncclSharpIflush,
  ncclSharpTest,
  ncclSharpCloseColl,
  ncclSharpCloseListen
};
