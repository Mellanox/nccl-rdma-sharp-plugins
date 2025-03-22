/*************************************************************************
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2016-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>

#define ENABLE_TIMER 0
#include "timer.h"
#include "p2p_plugin.h"
#include "core.h"
#include "socket.h"
#include "utils.h"
#include "param.h"

#include <assert.h>
#include "ibvwrap.h"

#define MAXNAMESIZE 64
static char ncclIbIfName[MAX_IF_NAME_SIZE+1];
static union ncclSocketAddress ncclIbIfAddr;

pthread_mutex_t ncclIbLock = PTHREAD_MUTEX_INITIALIZER;
int ncclIbRelaxedOrderingEnabled = 0;

static int ncclNMergedIbDevs = -1;
static int ncclNIbDevs = -1;

#define NCCL_IB_SL_DEFAULT 0
#define NCCL_IB_TC_DEFAULT 0

NCCL_PARAM(IbGidIndex, "IB_GID_INDEX", -1);
NCCL_PARAM(IbRoutableFlidIbGidIndex, "IB_ROUTABLE_FLID_GID_INDEX", 1);
NCCL_PARAM(IbRoceVersionNum, "IB_ROCE_VERSION_NUM", 2);
NCCL_PARAM(IbIsGlobal, "IB_IS_GLOBAL", 0);
NCCL_PARAM(IbTimeout, "IB_TIMEOUT", 20);
NCCL_PARAM(IbRetryCnt, "IB_RETRY_CNT", 7);
NCCL_PARAM(IbPkey, "IB_PKEY", 0);
NCCL_PARAM(IbUseInline, "IB_USE_INLINE", 0);
NCCL_PARAM(IbSl, "IB_SL", -1);
NCCL_PARAM(IbTc, "IB_TC", -1);
NCCL_PARAM(IbArThreshold, "IB_AR_THRESHOLD", 8192);
NCCL_PARAM(IbPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 2);
NCCL_PARAM(IbFifoTc, "IB_FIFO_TC", 0);
NCCL_PARAM(IbAsyncEvents,"IB_RETURN_ASYNC_EVENTS",1);
NCCL_PARAM(IbEceEnable,"IB_ECE_ENABLE",1);

static ncclResult_t ncclIbStatsCheckFatalCount(struct ncclIbStats* stat, const char* funcName) {
  if (ncclParamIbAsyncEvents() && __atomic_load_n(&stat->fatalErrorCount, __ATOMIC_RELAXED)) {
    WARN("communicator encountered a fatal error (detected in %s)\n", funcName);
    return ncclSystemError;
  }
  return ncclSuccess;
}

static pthread_t ncclIbAsyncThread;

// Determine whether RELAXED_ORDERING is enabled and possible
int ncclIbRelaxedOrderingCapable(void) {
  int roMode = ncclParamIbPciRelaxedOrdering();
  if (roMode == 1 || roMode == 2) {
    if (!IBV_ACCESS_RELAXED_ORDERING) {
      if(roMode == 1)
	WARN("NET/IB: PCI relaxed order memory access requested but not supported");
      return 0;
    }
  } else {
     return 0;
  }
  return 1;
}

static sa_family_t envIbAddrFamily(void) {
  sa_family_t family = AF_INET;
  const char* env = ncclGetEnv("NCCL_IB_ADDR_FAMILY");
  if (env == NULL || strlen(env) == 0) {
    return family;
  }

  INFO(NCCL_ENV, "NCCL_IB_ADDR_FAMILY set by environment to %s", env);

  if (strcmp(env, "AF_INET") == 0) {
    family = AF_INET;
  } else if (strcmp(env, "AF_INET6") == 0) {
    family = AF_INET6;
  }

  return family;
}

static void* envIbAddrRange(sa_family_t af, int* mask) {
  *mask = 0;
  static struct in_addr addr;
  static struct in6_addr addr6;
  void *ret = (af == AF_INET) ? (void *)&addr : (void *)&addr6;

  const char* env = ncclGetEnv("NCCL_IB_ADDR_RANGE");
  if (NULL == env || strlen(env) == 0) {
    return NULL;
  }

  INFO(NCCL_ENV, "NCCL_IB_ADDR_RANGE set by environment to %s", env);

  char addrString[128] = { 0 };
  snprintf(addrString, 128, "%s", env);
  char *addrStrPtr = addrString;
  char *maskStrPtr = strstr(addrString, "/");
  if (NULL == maskStrPtr) {
    return NULL;
  }
  *(maskStrPtr++) = '\0';

  if (inet_pton(af, addrStrPtr, ret) == 0) {
    INFO(NCCL_INIT|NCCL_NET, "NET/IB: Ip address '%s' is invalid for family %s, ignoring address", addrStrPtr, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    return NULL;
  }

  *mask = (int)strtol(maskStrPtr, NULL, 10);
  if (af == AF_INET && *mask > 32) {
    INFO(NCCL_INIT|NCCL_NET, "NET/IB: Ip address mask '%d' is invalid for family %s, ignoring mask", *mask, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    *mask = 0;
    ret = NULL;
  } else if (af == AF_INET6 && *mask > 128) {
    INFO(NCCL_INIT|NCCL_NET, "NET/IB: Ip address mask '%d' is invalid for family %s, ignoring mask", *mask, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    *mask = 0;
    ret = NULL;
  }

  return ret;
}

static sa_family_t getGidAddrFamily(union ibv_gid* gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  bool isIpV4Mapped = ((a->s6_addr32[0] | a->s6_addr32[1]) | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL;
  bool isIpV4MappedMulticast = (a->s6_addr32[0] == htonl(0xff0e0000) && ((a->s6_addr32[1] | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL));
  return (isIpV4Mapped || isIpV4MappedMulticast) ? AF_INET : AF_INET6;
}

static bool matchGidAddrPrefix(sa_family_t af, void* prefix, int prefixlen, union ibv_gid* gid) {
  struct in_addr *base = NULL;
  struct in6_addr *base6 = NULL;
  struct in6_addr *addr6 = NULL;;
  if (af == AF_INET) {
    base = (struct in_addr *)prefix;
  } else {
    base6 = (struct in6_addr *)prefix;
  }
  addr6 = (struct in6_addr *)gid->raw;

#define NETMASK(bits) (htonl(0xffffffff ^ ((1 << (32 - bits)) - 1)))

  int i = 0;
  while (prefixlen > 0 && i < 4) {
    if (af == AF_INET) {
      int mask = NETMASK(prefixlen);
      if ((base->s_addr & mask) ^ (addr6->s6_addr32[3] & mask)) {
        break;
      }
      prefixlen = 0;
      break;
    } else {
      if (prefixlen >= 32) {
        if (base6->s6_addr32[i] ^ addr6->s6_addr32[i]) {
          break;
        }
        prefixlen -= 32;
        ++i;
      } else {
        int mask = NETMASK(prefixlen);
        if ((base6->s6_addr32[i] & mask) ^ (addr6->s6_addr32[i] & mask)) {
          break;
        }
        prefixlen = 0;
      }
    }
  }

  return (prefixlen == 0) ? true : false;
}

static bool configuredGid(union ibv_gid* gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  int trailer = (a->s6_addr32[1] | a->s6_addr32[2] | a->s6_addr32[3]);
  if (((a->s6_addr32[0] | trailer) == 0UL) || ((a->s6_addr32[0] == htonl(0xfe800000)) && (trailer == 0UL))) {
    return false;
  }
  return true;
}

static bool linkLocalGid(union ibv_gid* gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  if (a->s6_addr32[0] == htonl(0xfe800000) && a->s6_addr32[1] == 0UL) {
    return true;
  }
  return false;
}

static bool validGid(union ibv_gid* gid) {
  return (configuredGid(gid) && !linkLocalGid(gid));
}

static ncclResult_t ncclIbRoceGetVersionNum(const char* deviceName, int portNum, int gidIndex, int* version) {
  char gidRoceVerStr[16] = { 0 };
  char roceTypePath[PATH_MAX] = { 0 };
  snprintf(roceTypePath, sizeof(roceTypePath), "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d", deviceName, portNum, gidIndex);

  int fd = open(roceTypePath, O_RDONLY);
  if (fd == -1) {
    WARN("NET/IB: open failed in ncclIbRoceGetVersionNum: %s", strerror(errno));
    return ncclSystemError;
  }
  int ret = read(fd, gidRoceVerStr, 15);
  close(fd);

  if (ret == -1) {
    // In containerized environments, read could return EINVAL if the GID index is not mapped to the
    // container sysfs. In this case return nc
    if (errno == EINVAL) return ncclSuccess;
    WARN("NET/IB: read failed in ncclIbRoceGetVersionNum: %s", strerror(errno));
    return ncclSystemError;
  }

  if (strlen(gidRoceVerStr)) {
    if (strncmp(gidRoceVerStr, "IB/RoCE v1", strlen("IB/RoCE v1")) == 0 || strncmp(gidRoceVerStr, "RoCE v1", strlen("RoCE v1")) == 0) {
      *version = 1;
    } else if (strncmp(gidRoceVerStr, "RoCE v2", strlen("RoCE v2")) == 0) {
      *version = 2;
    }
  }

  return ncclSuccess;
}

static ncclResult_t ncclUpdateGidIndex(struct ibv_context* context, uint8_t portNum, sa_family_t af, void* prefix, int prefixlen, int roceVer, int gidIndexCandidate, int* gidIndex) {
  union ibv_gid gid, gidCandidate;
  NCCLCHECK(wrap_ibv_query_gid(context, portNum, *gidIndex, &gid));
  NCCLCHECK(wrap_ibv_query_gid(context, portNum, gidIndexCandidate, &gidCandidate));

  sa_family_t usrFam = af;
  sa_family_t gidFam = getGidAddrFamily(&gid);
  sa_family_t gidCandidateFam = getGidAddrFamily(&gidCandidate);
  bool gidCandidateMatchSubnet = matchGidAddrPrefix(usrFam, prefix, prefixlen, &gidCandidate);

  if (gidCandidateFam != gidFam && gidCandidateFam == usrFam && gidCandidateMatchSubnet) {
    *gidIndex = gidIndexCandidate;
  } else {
    if (gidCandidateFam != usrFam || !validGid(&gidCandidate) || !gidCandidateMatchSubnet) {
      return ncclSuccess;
    }
    int usrRoceVer = roceVer;
    int gidRoceVerNum, gidRoceVerNumCandidate = -1;
    const char* deviceName = wrap_ibv_get_device_name(context->device);
    NCCLCHECK(ncclIbRoceGetVersionNum(deviceName, portNum, *gidIndex, &gidRoceVerNum));
    NCCLCHECK(ncclIbRoceGetVersionNum(deviceName, portNum, gidIndexCandidate, &gidRoceVerNumCandidate));
    if ((gidRoceVerNum != gidRoceVerNumCandidate || !validGid(&gid)) && gidRoceVerNumCandidate == usrRoceVer) {
      *gidIndex = gidIndexCandidate;
    }
  }

  return ncclSuccess;
}

// GID Format
// global:  |              64b  - subnet-prefix                |                 64b - EUI                          |
// raw   :  | 10b fixed | 22b 0 | 16b FLID | 16b subnet-prefix |                 64b - EUI                          |
static uint16_t ncclIbExtractLocalSubnetPrefix(uint64_t subnet_prefix)
{
  return (be64toh(subnet_prefix) & 0xffff);
}

static int ncclIbExtractFlid (union ibv_gid *gid)
{
  return ntohs(*((uint16_t*)((uintptr_t)(gid->raw) + 4)));
}

static ncclResult_t ncclIbGetGidIndex(struct ibv_context *context, uint8_t portNum, struct ibv_port_attr* portAttr, int *gidIndex) {
  int gidTblLen = portAttr->gid_tbl_len;

  //for IB, choose GID Index that will have routable FLID if present
  if (portAttr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
    union ibv_gid gid;
    int routableGidIndex = ncclParamIbRoutableFlidIbGidIndex();
    if (routableGidIndex < gidTblLen) {
      NCCLCHECK(wrap_ibv_query_gid(context, portNum, routableGidIndex, &gid));
      if (ncclIbExtractFlid(&gid) != 0) {
        *gidIndex = routableGidIndex;
        return ncclSuccess;
      }
    }
    *gidIndex = 0;
    return ncclSuccess;
  }

  //for ROCE
  *gidIndex = ncclParamIbGidIndex();
  if (*gidIndex >= 0) {
    return ncclSuccess;
  }

  sa_family_t userAddrFamily = envIbAddrFamily();
  int userRoceVersion = ncclParamIbRoceVersionNum();
  int prefixlen;
  void *prefix = envIbAddrRange(userAddrFamily, &prefixlen);

  *gidIndex = 0;
  for (int gidIndexNext = 1; gidIndexNext < gidTblLen; ++gidIndexNext) {
    NCCLCHECK(ncclUpdateGidIndex(context, portNum, userAddrFamily, prefix, prefixlen, userRoceVersion, gidIndexNext, gidIndex));
  }

  return ncclSuccess;
}


NCCL_PARAM(IbDisable, "IBEXT_DISABLE", 0);
NCCL_PARAM(IbMergeVfs, "IB_MERGE_VFS", 1);
NCCL_PARAM(IbMergeNics, "IB_MERGE_NICS", 1);

extern ncclDebugLogger_t pluginLogFunction;

ncclResult_t ncclIbDevices(int* ndev) {
  *ndev = ncclNMergedIbDevs;
  return ncclSuccess;
}

ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props)
{
  return nccl_p2p_ib_get_properties(ncclIbDevs, ncclNMergedIbDevs, dev, props);
}


ncclResult_t ncclIbGetProperties_v9(int dev, ncclNetProperties_v9_t* props_v9) {
  ncclNetProperties_t props;
  ncclResult_t ret = nccl_p2p_ib_get_properties(ncclIbDevs, ncclNMergedIbDevs, dev, &props);
  if (ret != ncclSuccess) return ret;
  props_v9->name = props.name;
  props_v9->pciPath = props.pciPath;
  props_v9->guid = props.guid;
  props_v9->ptrSupport = props.ptrSupport;
  props_v9->regIsGlobal = props.regIsGlobal;
  props_v9->forceFlush = props.forceFlush;
  props_v9->speed = props.speed;
  props_v9->port = props.port;
  props_v9->latency = props.latency;
  props_v9->maxComms = props.maxComms;
  props_v9->maxRecvs = props.maxRecvs;
  props_v9->netDeviceType = props.netDeviceType;
  props_v9->netDeviceVersion = props.netDeviceVersion;
  props_v9->maxP2pBytes = props.maxP2pBytes;
  props_v9->maxCollBytes = props.maxCollBytes;
  props_v9->vProps.ndevs = props.vProps.ndevs;
  for(int i=0; i<props.vProps.ndevs; i++) {
    props_v9->vProps.devs[i] = props.vProps.devs[i];
  }
  return ncclSuccess;
}

ncclResult_t ncclIbGetProperties_v8(int dev, ncclNetProperties_v8_t* props_v8)
{
  ncclNetProperties_t props;
  ncclResult_t ret = nccl_p2p_ib_get_properties(ncclIbDevs, ncclNMergedIbDevs, dev, &props);
  if (ret != ncclSuccess) return ret;
  props_v8->name = props.name;
  props_v8->pciPath = props.pciPath;
  props_v8->guid = props.guid;
  props_v8->ptrSupport = props.ptrSupport;
  props_v8->regIsGlobal = props.regIsGlobal;
  props_v8->speed = props.speed;
  props_v8->latency = props.latency;
  props_v8->port = props.port;
  props_v8->maxComms = props.maxComms;
  props_v8->maxRecvs = props.maxRecvs;
  props_v8->netDeviceType = props.netDeviceType;
  props_v8->netDeviceVersion = props.netDeviceVersion;
  return ncclSuccess;
}

ncclResult_t ncclIbGetProperties_v7(int dev, ncclNetProperties_v7_t* props_v7)
{
  ncclNetProperties_t props;
  ncclResult_t ret = nccl_p2p_ib_get_properties(ncclIbDevs, ncclNMergedIbDevs, dev, &props);
  if (ret != ncclSuccess) return ret;
  props_v7->name = props.name;
  props_v7->pciPath = props.pciPath;
  props_v7->guid = props.guid;
  props_v7->ptrSupport = props.ptrSupport;
  props_v7->speed = props.speed;
  props_v7->latency = props.latency;
  props_v7->port = props.port;
  props_v7->maxComms = props.maxComms;
  props_v7->maxRecvs = props.maxRecvs;
  props_v7->netDeviceType = props.netDeviceType;
  props_v7->netDeviceVersion = props.netDeviceVersion;
  return ncclSuccess;
}

ncclResult_t ncclIbGetProperties_v6(int dev, ncclNetProperties_v6_t* props_v6)
{
  ncclNetProperties_t props;
  ncclResult_t ret = nccl_p2p_ib_get_properties(ncclIbDevs, ncclNMergedIbDevs, dev, &props);
  if (ret != ncclSuccess) return ret;
  props_v6->name = props.name;
  props_v6->pciPath = props.pciPath;
  props_v6->guid = props.guid;
  props_v6->ptrSupport = props.ptrSupport;
  props_v6->speed = props.speed;
  props_v6->latency = props.latency;
  props_v6->port = props.port;
  props_v6->maxComms = props.maxComms;
  props_v6->maxRecvs = props.maxRecvs;

  return ncclSuccess;
};

#define NCCL_IB_MAX_QPS 128

struct ncclIbQpInfo {
  uint32_t qpn;

  // Fields needed for ece (enhanced connection establishment)
  struct ibv_ece ece;
  int ece_supported;
  int devIndex;
};


// Per-Dev connection metadata
typedef struct ncclIbDevInfo {
  uint32_t lid;
  uint8_t ib_port;
  enum ibv_mtu mtu;
  uint8_t link_layer;
  uint8_t is_global;

  //For RoCE and IB GRH & Rounter
  union ibv_gid gid;

  // FIFO RDMA info
  uint32_t fifoRkey;

  //remote dev info
  union ibv_gid remoteGid;
} ncclIbDevInfo;


// Struct containing everything needed to establish connections
typedef struct ncclIbConnectionMetadata {
  struct ncclIbQpInfo qpInfo[NCCL_IB_MAX_QPS];
  struct ncclIbDevInfo devs[NCCL_IB_MAX_DEVS_PER_NIC];
  char devName[MAX_MERGED_DEV_NAME];
  uint64_t fifoAddr;
  int ndevs;
  int tc;
  int sl;
} ncclIbConnectionMetadata;

enum ncclIbCommState {
  ncclIbCommStateStart = 0,
  ncclIbCommStateConnect = 1,
  ncclIbCommStateAccept = 3,
  ncclIbCommStateSend = 4,
  ncclIbCommStateRecv = 5,
  ncclIbCommStateConnecting = 6,
  ncclIbCommStateConnected = 7,
  ncclIbCommStatePendingReady = 8,
  ncclIbCommStateSendDevList = 9,
  ncclIbCommStateRecvDevList = 10,
};

struct ncclIbCommStage {
  enum ncclIbCommState state;
  int offset;
  void* buffer;
  void* comm;
};


struct ncclIbHandle {
  union ncclSocketAddress connectAddr; // Filled by the target
  uint64_t magic; // random number to help debugging
  struct ncclIbCommStage stage; // Used by the other side when connecting
};



#define NCCL_NET_IB_REQ_UNUSED 0
#define NCCL_NET_IB_REQ_SEND 1
#define NCCL_NET_IB_REQ_RECV 2
#define NCCL_NET_IB_REQ_FLUSH 3
const char* reqTypeStr[] = { "Unused", "Send", "Recv", "Flush" };

struct ncclIbListenComm {
  int dev;
  struct ncclSocket sock;
  struct ncclIbCommStage stage;
};

struct ncclIbSendFifo {
  uint64_t addr;
  uint64_t size;
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t nreqs;
  uint32_t tag;
  uint64_t idx;
  char padding[16];
};

typedef struct ncclIbQp {
  struct ibv_qp* qp;
  int devIndex;
  int remDevIdx;
} ncclIbQp;

struct ncclIbRemSizesFifo {
  int elems[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t fifoTail;
  uint64_t addr;
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t flags;
  struct ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ibv_sge sge;
};

// A per-dev struct for netIbSendComm
typedef struct ncclIbSendCommDev {
  struct ncclIbNetCommDevBase base;
  struct ibv_mr* fifoMr;
} __attribute__((aligned(8))) ncclIbSendCommDev;


// Wrapper to track an MR per-device, if needed
struct ncclIbMrHandle {
  struct ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC];
};

typedef struct ncclIbNetCommBase {
  ncclNetVDeviceProps_t vProps;
  bool isSend;
  struct ncclIbRequest reqs[MAX_REQUESTS];
  struct ncclIbQp qps[NCCL_IB_MAX_QPS];
  int nqps;
  int qpIndex;
  int devIndex;
  struct ncclSocket sock;
  int ready;
  // Track necessary remDevInfo here
  int nRemDevs;
  int nDataQps;
  struct ncclIbDevInfo remDevs[NCCL_IB_MAX_DEVS_PER_NIC];
  // statistics about the comm
  struct ncclIbStats stats;
 }  __attribute__((aligned(32))) ncclIbNetCommBase;

struct ncclIbSendComm {
  struct ncclIbNetCommBase base;
  // Start with fifo and ibv structs as they have alignment restrictions
  struct ncclIbSendFifo fifo[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  struct ibv_sge sges[NCCL_NET_IB_MAX_RECVS];
  struct ibv_send_wr wrs[NCCL_NET_IB_MAX_RECVS + 1];
  // Each dev correlates to a mergedIbDev
  struct ncclIbSendCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbRequest* fifoReqs[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  struct ncclIbRemSizesFifo remSizesFifo;
  uint64_t fifoHead;
  int ar; // Use adaptive routing when all merged devices have it enabled
};
struct ncclIbGpuFlush {
  struct ibv_mr* hostMr;
  struct ibv_sge sge;
  struct ncclIbQp qp;
};

struct ncclIbRemFifo {
  struct ncclIbSendFifo elems[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t fifoTail;
  uint64_t addr;
  uint32_t flags;
};

struct ncclIbRecvCommDev {
  struct ncclIbNetCommDevBase base;
  struct ncclIbGpuFlush gpuFlush;
  struct ibv_mr* fifoMr;
  struct ibv_sge fifoSge;
  struct ibv_mr* sizesFifoMr;
} __attribute__((aligned(16)));

struct ncclIbRecvComm {
  struct ncclIbNetCommBase base;
  struct ncclIbRecvCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  struct ncclIbRemFifo remFifo;
  int sizesFifo[MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  int gpuFlushHostMem;
  int flushEnabled;
};

ncclResult_t ncclIbInit(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  if (ncclParamIbDisable()) return ncclInternalError;

  // The SendFifo needs to be 32-byte aligned and each element needs
  // to be a 32-byte multiple, so that an entry does not get split and
  // written out of order when IB Relaxed Ordering is enabled
  NCCL_STATIC_ASSERT((sizeof(struct ncclIbNetCommBase) % 32) == 0, "ncclIbNetCommBase size must be 32-byte multiple to ensure fifo is at proper offset");
  NCCL_STATIC_ASSERT((offsetof(struct ncclIbSendComm, fifo) % 32) == 0, "ncclIbSendComm fifo must be 32-byte aligned");
  NCCL_STATIC_ASSERT((sizeof(struct ncclIbSendFifo) % 32) == 0, "ncclIbSendFifo element size must be 32-byte multiples");
  NCCL_STATIC_ASSERT((offsetof(struct ncclIbSendComm, sges) % 32) == 0, "sges must be 32-byte aligned");
  NCCL_STATIC_ASSERT((offsetof(struct ncclIbSendComm, wrs) % 32) == 0, "wrs must be 32-byte aligned");
  NCCL_STATIC_ASSERT((offsetof(struct ncclIbRecvComm, remFifo) % 32) == 0, "ncclIbSendComm fifo must be 32-byte aligned");


  return nccl_p2p_ib_init(&ncclNIbDevs, &ncclNMergedIbDevs, ncclIbDevs, ncclIbIfName, &ncclIbIfAddr, &ncclIbAsyncThread, logFunction);
}

NCCL_PARAM(IbQpsPerConn, "IB_QPS_PER_CONNECTION", 1);

static void ncclIbAddEvent(struct ncclIbRequest* req, int devIndex, struct ncclIbNetCommDevBase* base) {
  req->events[devIndex]++;
  req->devBases[devIndex] = base;
}

ncclResult_t ncclIbInitCommDevBase(int ibDevN, struct ncclIbNetCommDevBase* base, void* cq_context) {
  base->ibDevN = ibDevN;
  ncclIbDev* ibDev = ncclIbDevs + ibDevN;
  pthread_mutex_lock(&ibDev->lock);
  if (0 == ibDev->pdRefs++) {
    ncclResult_t res;
    NCCLCHECKGOTO(wrap_ibv_alloc_pd(&ibDev->pd, ibDev->context), res, failure);
    if (0) {
    failure:
      pthread_mutex_unlock(&ibDev->lock);
      return res;
    }
  }
  base->pd = ibDev->pd;
  pthread_mutex_unlock(&ibDev->lock);

  // Recv requests can generate 2 completions (one for the post FIFO, one for the Recv).
  NCCLCHECK(wrap_ibv_create_cq(&base->cq, ibDev->context, 2*MAX_REQUESTS*ncclParamIbQpsPerConn(), cq_context, NULL, 0));

  return ncclSuccess;
}

ncclResult_t ncclIbDestroyBase(struct ncclIbNetCommDevBase* base) {
  ncclResult_t res;
  NCCLCHECK(wrap_ibv_destroy_cq(base->cq));

  pthread_mutex_lock(&ncclIbDevs[base->ibDevN].lock);
  if (0 == --ncclIbDevs[base->ibDevN].pdRefs) {
    NCCLCHECKGOTO(wrap_ibv_dealloc_pd(ncclIbDevs[base->ibDevN].pd), res, returning);
  }
  res = ncclSuccess;
returning:
  pthread_mutex_unlock(&ncclIbDevs[base->ibDevN].lock);
  return res;
}

ncclResult_t ncclIbCreateQp(uint8_t ib_port, struct ncclIbNetCommDevBase* base, int access_flags, void* qp_context, struct ncclIbQp* qp) {
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.qp_context = qp_context;
  qpInitAttr.send_cq = base->cq;
  qpInitAttr.recv_cq = base->cq;
  qpInitAttr.qp_type = IBV_QPT_RC;
  // We might send 2 messages per send (RDMA and RDMA_WITH_IMM)
  qpInitAttr.cap.max_send_wr = 2*MAX_REQUESTS;
  qpInitAttr.cap.max_recv_wr = MAX_REQUESTS;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = ncclParamIbUseInline() ? sizeof(struct ncclIbSendFifo) : 0;
  NCCLCHECK(wrap_ibv_create_qp(&qp->qp, base->pd, &qpInitAttr));
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = ncclParamIbPkey();
  qpAttr.port_num = ib_port;
  qpAttr.qp_access_flags = access_flags;
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  TRACE(NCCL_NET, "NET/IB : ncclIbCreateQp port=%d dev=%d devName=%s ndevs=%d nmdevs=%d qpn=%u pkey=%u pd=%p",
    ib_port, base->ibDevN, ncclIbDevs[base->ibDevN].devName, ncclNIbDevs, ncclNMergedIbDevs, qp->qp->qp_num, qpAttr.pkey_index, base->pd);
  return ncclSuccess;
}

ncclResult_t ncclIbRtrQp(struct ibv_qp* qp, struct ncclIbGidInfo* sGidInfo, uint32_t dest_qp_num, struct ncclIbDevInfo* info, bool fifoTc, int tc, int sl) {
  struct ibv_qp_attr qpAttr;
  int same_subnet;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = info->mtu;
  qpAttr.dest_qp_num = dest_qp_num;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;
  if (info->link_layer == IBV_LINK_LAYER_ETHERNET) {
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.dgid.global.subnet_prefix = info->gid.global.subnet_prefix;
    qpAttr.ah_attr.grh.dgid.global.interface_id = info->gid.global.interface_id;
    qpAttr.ah_attr.grh.flow_label = 0;
    qpAttr.ah_attr.grh.sgid_index = sGidInfo->localGidIndex;
    qpAttr.ah_attr.grh.hop_limit = 255;
    qpAttr.ah_attr.grh.traffic_class = fifoTc && ncclParamIbFifoTc() != -1 ? ncclParamIbFifoTc() : tc;
  } else {
    same_subnet = (ncclIbExtractLocalSubnetPrefix(sGidInfo->localGid.global.subnet_prefix) ==
                   ncclIbExtractLocalSubnetPrefix(info->gid.global.subnet_prefix));
    qpAttr.ah_attr.is_global = 0;
    qpAttr.ah_attr.dlid = info->lid;
    if (!same_subnet || info->is_global) {
      if (!same_subnet) {
        uint16_t flid = ncclIbExtractFlid(&info->gid);
        if (flid == 0) {
          WARN("Warning: remote FLID configured as zero even when endpoints are on different subnets, using dlid as fallback");
          qpAttr.ah_attr.dlid = info->lid;
        } else {
          qpAttr.ah_attr.dlid = ncclIbExtractFlid(&info->gid);
        }
      }
      qpAttr.ah_attr.is_global = 1;
      qpAttr.ah_attr.grh.dgid.global.subnet_prefix = info->gid.global.subnet_prefix;
      qpAttr.ah_attr.grh.dgid.global.interface_id = info->gid.global.interface_id;
      qpAttr.ah_attr.grh.sgid_index = sGidInfo->localGidIndex;
      qpAttr.ah_attr.grh.hop_limit = 255;
    }
  }
  qpAttr.ah_attr.sl = sl;
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = info->ib_port;
  TRACE(NCCL_NET, "NET/IB : ncclIbRtrQp qpn=%u mtu=%d dst=%u ll=%u port=%u sl: %d tc: %d", qp->qp_num, info->mtu, dest_qp_num, info->link_layer, info->ib_port, qpAttr.ah_attr.sl, qpAttr.ah_attr.grh.traffic_class);
  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));
  return ncclSuccess;
}

ncclResult_t ncclIbRtsQp(struct ibv_qp* qp) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = ncclParamIbTimeout();
  qpAttr.retry_cnt = ncclParamIbRetryCnt();
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
  return ncclSuccess;
}

ncclResult_t ncclIbListen(int dev, void* opaqueHandle, void** listenComm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbListenComm* comm;
  comm = malloc(sizeof(struct ncclIbListenComm));
  memset(comm, 0, sizeof(struct ncclIbListenComm));
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  NCCL_STATIC_ASSERT(sizeof(struct ncclIbHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclIbHandle size too large");
  memset(handle, 0, sizeof(struct ncclIbHandle));
  comm->dev = dev;
  handle->magic = NCCL_SOCKET_MAGIC;
  NCCLCHECKGOTO(ncclSocketInit(&comm->sock, &ncclIbIfAddr, handle->magic, ncclSocketTypeNetIb, NULL, 1, 0), ret, fail);
  NCCLCHECKGOTO(ncclSocketListen(&comm->sock), ret, fail);
  NCCLCHECKGOTO(ncclSocketGetAddr(&comm->sock, &handle->connectAddr), ret, fail);
  *listenComm = comm;
exit:
  return ret;
fail:
  (void)ncclSocketClose(&comm->sock, false);
  free(comm);
  goto exit;
}

ncclResult_t ncclIbConnect(int dev, ncclNetCommConfig_t* config, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  struct ncclIbCommStage* stage = &handle->stage;
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)stage->comm;
  struct ncclIbConnectionMetadata remMeta;
  int ready;
  uint8_t link_layer = IBV_LINK_LAYER_UNSPECIFIED;
  *sendComm = NULL;

  if (stage->state == ncclIbCommStateConnect)      goto ib_connect_check;
  if (stage->state == ncclIbCommStateSendDevList)  goto ib_send_dev_list;
  if (stage->state == ncclIbCommStateRecvDevList)  goto ib_recv_dev_list;
  if (stage->state == ncclIbCommStateSend)         goto ib_send;
  if (stage->state == ncclIbCommStateConnecting)   goto ib_connect;
  if (stage->state == ncclIbCommStateConnected)    goto ib_send_ready;
  if (stage->state != ncclIbCommStateStart) {
    WARN("Error: trying to connect already connected sendComm");
    return ncclInternalError;
  }
  stage->buffer = NULL;

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(struct ncclIbSendComm)));
  NCCLCHECKGOTO(ncclIbStatsInit(&comm->base.stats), ret, fail);
  NCCLCHECKGOTO(ncclSocketInit(&comm->base.sock, &handle->connectAddr, handle->magic, ncclSocketTypeNetIb, NULL, 1, 0), ret, fail);
  stage->comm = comm;
  stage->state = ncclIbCommStateConnect;
  NCCLCHECKGOTO(ncclSocketConnect(&comm->base.sock), ret, fail);

ib_connect_check:
  /* since ncclSocketConnect is async, we must check if connection is complete */
  NCCLCHECKGOTO(ncclSocketReady(&comm->base.sock, &ready), ret, fail);
  if (!ready) return ncclSuccess;

  // IB Setup
  struct ncclIbMergedDev* mergedDev;
  if (dev >= ncclNMergedIbDevs) {
    WARN("NET/IB : Trying to use non-existent virtual device %d", dev);
    return ncclInternalError;
  }

  mergedDev = ncclIbMergedDevs + dev;
  comm->base.vProps = mergedDev->vProps;
  comm->base.isSend = true;
  stage->state = ncclIbCommStateSendDevList;
  stage->offset = 0;
  struct ncclIbConnectionMetadata meta;
  NCCLCHECKGOTO(ncclIbMalloc((void**)&stage->buffer, sizeof(meta)), ret, fail);
  memcpy(stage->buffer, &mergedDev->vProps, sizeof(ncclNetVDeviceProps_t));

// In the case of mismatched nDevs, we will make sure that both sides of a logical connection have the same number of RC qps
ib_send_dev_list:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, stage->buffer, sizeof(ncclNetVDeviceProps_t), &stage->offset, NULL));
  if (stage->offset != sizeof(ncclNetVDeviceProps_t)) return ncclSuccess;

  stage->state = ncclIbCommStateRecvDevList;
  stage->offset = 0;

ib_recv_dev_list:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, stage->buffer, sizeof(ncclNetVDeviceProps_t), &stage->offset, NULL));
  if (stage->offset != sizeof(ncclNetVDeviceProps_t)) return ncclSuccess;
  stage->offset = 0;
  ncclNetVDeviceProps_t remoteVProps;
  memcpy(&remoteVProps, stage->buffer, sizeof(ncclNetVDeviceProps_t));
  mergedDev = ncclIbMergedDevs + dev;
  comm->base.vProps = mergedDev->vProps;
  int localNqps, remoteNqps;
  localNqps  = ncclParamIbQpsPerConn() * comm->base.vProps.ndevs; // We must have at least 1 qp per-device
  remoteNqps = ncclParamIbQpsPerConn() * remoteVProps.ndevs;
  comm->base.nqps = remoteNqps > localNqps ? remoteNqps : localNqps; // Select max nqps (local or remote)

  // Init PD, Ctx for each IB device
  comm->ar = 1; // Set to 1 for logic
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    int ibDevN = comm->base.vProps.devs[i];
    NCCLCHECKGOTO(ncclIbInitCommDevBase(ibDevN, &comm->devs[i].base, &comm->base.stats), ret, fail);
    comm->ar = comm->ar && ncclIbDevs[ibDevN].ar; // ADAPTIVE_ROUTING - if all merged devs have it enabled
  }

  memset(&meta, 0, sizeof(meta));
  meta.ndevs = comm->base.vProps.ndevs;

  // Alternate QPs between devices
  int devIndex;
  devIndex = 0;
  for (int q = 0; q < comm->base.nqps; q++) {
    ncclIbSendCommDev* commDev = comm->devs + devIndex;
    ncclIbDev* ibDev = ncclIbDevs + commDev->base.ibDevN;
    NCCLCHECKGOTO(ncclIbCreateQp(ibDev->portNum, &commDev->base, IBV_ACCESS_REMOTE_WRITE, &comm->base.stats, comm->base.qps + q), ret, fail);
    comm->base.qps[q].devIndex = devIndex;
    meta.qpInfo[q].qpn      = comm->base.qps[q].qp->qp_num;
    meta.qpInfo[q].devIndex = comm->base.qps[q].devIndex;

    if (ncclParamIbEceEnable()) {
      // Query ece capabilities (enhanced connection establishment)
      NCCLCHECKGOTO(wrap_ibv_query_ece(comm->base.qps[q].qp, &meta.qpInfo[q].ece, &meta.qpInfo[q].ece_supported), ret, fail);
    } else {
      meta.qpInfo[q].ece_supported = 0;
    }
    devIndex = (devIndex + 1) % comm->base.vProps.ndevs;
  }

  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    ncclIbSendCommDev* commDev = comm->devs + i;
    ncclIbDev* ibDev = ncclIbDevs + commDev->base.ibDevN;

    // Write to the metadata struct via this pointer
    ncclIbDevInfo* devInfo = meta.devs + i;
    devInfo->ib_port       = ibDev->portNum;
    devInfo->mtu           = ibDev->portAttr.active_mtu;
    devInfo->lid           = ibDev->portAttr.lid;

    // Prepare my fifo
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->fifoMr, commDev->base.pd, comm->fifo, sizeof(struct ncclIbSendFifo)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
    devInfo->fifoRkey = commDev->fifoMr->rkey;

    devInfo->is_global = (ncclParamIbIsGlobal()
    #if HAVE_DECL_IBV_QPF_GRH_REQUIRED
                          || (ibDev->portAttr.flags & IBV_QPF_GRH_REQUIRED)
#endif
    );
    
    // Pack local GID info
    devInfo->link_layer = commDev->base.gidInfo.link_layer = ibDev->portAttr.link_layer;
    NCCLCHECKGOTO(ncclIbGetGidIndex(ibDev->context, ibDev->portNum, &ibDev->portAttr, &commDev->base.gidInfo.localGidIndex), ret, fail);
    NCCLCHECKGOTO(wrap_ibv_query_gid(ibDev->context, ibDev->portNum, commDev->base.gidInfo.localGidIndex, &commDev->base.gidInfo.localGid), ret, fail);
    devInfo->gid.global.subnet_prefix = commDev->base.gidInfo.localGid.global.subnet_prefix;
    devInfo->gid.global.interface_id = commDev->base.gidInfo.localGid.global.interface_id;

    // info logging
    if (devInfo->link_layer == IBV_LINK_LAYER_INFINIBAND) { // IB
      for (int q = 0; q < comm->base.nqps; q++) {
        // Print just the QPs for this dev
        if (comm->base.qps[q].devIndex == i)
          INFO(NCCL_NET,"NET/IB: %s %d IbDev %d Port %d qpn %d mtu %d LID %d subnet-prefix %lu  FLID %d fifoRkey=0x%x fifoLkey=0x%x",
            comm->base.vProps.ndevs > 2 ? "NCCL MergedDev" : "NCCL Dev",
            dev, commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn, devInfo->mtu, devInfo->lid,
	    devInfo->gid.global.subnet_prefix, ncclIbExtractFlid(&devInfo->gid), devInfo->fifoRkey, commDev->fifoMr->lkey);
      }
    } else { // RoCE
      for (int q = 0; q < comm->base.nqps; q++) {
        // Print just the QPs for this dev
        if (comm->base.qps[q].devIndex == i)
          INFO(NCCL_NET,"NET/IB: %s %d IbDev %d Port %d qpn %d mtu %d query_ece={supported=%d, vendor_id=0x%x, options=0x%x, comp_mask=0x%x} GID %ld (%lX/%lX) fifoRkey=0x%x fifoLkey=0x%x",
          comm->base.vProps.ndevs > 2 ? "NCCL MergedDev" : "NCCL Dev", dev,
          commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn, devInfo->mtu, meta.qpInfo[q].ece_supported, meta.qpInfo[q].ece.vendor_id, meta.qpInfo[q].ece.options, meta.qpInfo[q].ece.comp_mask, (int64_t)commDev->base.gidInfo.localGidIndex,
          devInfo->gid.global.subnet_prefix, devInfo->gid.global.interface_id, devInfo->fifoRkey, commDev->fifoMr->lkey);
      }
    }
    if (link_layer == IBV_LINK_LAYER_UNSPECIFIED) link_layer = devInfo->link_layer;
    if (link_layer != devInfo->link_layer) {
      int ibDev0 = comm->devs[0].base.ibDevN;
      WARN("NET/IB : Attempted to connect incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
           commDev->base.ibDevN, ibDev->devName, ibDev->portNum, NCCL_IB_LLSTR(ibDev->portAttr.link_layer), ibDev0, ncclIbDevs[ibDev0].devName, ncclIbDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
      return ncclInternalError;
    }
  }

  meta.fifoAddr = (uint64_t)comm->fifo;
  meta.sl = (ncclParamIbSl() != -1) ? ncclParamIbSl() : (config && config->trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF) ? config->trafficClass : NCCL_IB_SL_DEFAULT;
  meta.tc = (ncclParamIbTc() != -1) ? ncclParamIbTc() : (config && config->trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF) ? config->trafficClass : NCCL_IB_TC_DEFAULT;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  stage->state = ncclIbCommStateSend;
  stage->offset = 0;

  memcpy(stage->buffer, &meta, sizeof(meta));

ib_send:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, stage->buffer, sizeof(meta), &stage->offset, NULL), ret, fail);
  if (stage->offset != sizeof(meta)) return ncclSuccess;


  stage->state = ncclIbCommStateConnecting;
  stage->offset = 0;
  // Clear the staging buffer for re-use
  memset(stage->buffer, 0, sizeof(meta));

ib_connect:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, stage->buffer, sizeof(ncclIbConnectionMetadata), &stage->offset, NULL), ret, fail);
  if (stage->offset != sizeof(remMeta)) return ncclSuccess;

  memcpy(&remMeta, stage->buffer, sizeof(ncclIbConnectionMetadata));

  comm->base.nRemDevs = remMeta.ndevs;

  // ensure that the remote devices have the same link layer than the local devices used in the connection.
  if (comm->base.vProps.ndevs > 0) {
    int ibDev0 = comm->devs[0].base.ibDevN;
    link_layer = ncclIbDevs[ibDev0].portAttr.link_layer;
    for (int i = 0; i < remMeta.ndevs; i++) {
      if (remMeta.devs[i].link_layer != link_layer) {
        WARN("NET/IB : Remote %s device is incompatible with the local [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
             NCCL_IB_LLSTR(remMeta.devs[i].link_layer), ibDev0, ncclIbDevs[ibDev0].devName, ncclIbDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
        return ncclInternalError;
      }
    }
  }

  // Copy remDevInfo for things like remGidInfo, remFifoAddr, etc.
  for (int i = 0; i < remMeta.ndevs; i++) {
    comm->base.remDevs[i] = remMeta.devs[i];
    comm->base.remDevs[i].remoteGid.global.interface_id = comm->base.remDevs[i].gid.global.interface_id;
    comm->base.remDevs[i].remoteGid.global.subnet_prefix = comm->base.remDevs[i].gid.global.subnet_prefix;

    // Retain remote sizes fifo info and prepare RDMA ops
    comm->remSizesFifo.rkeys[i] = remMeta.devs[i].fifoRkey;
    comm->remSizesFifo.addr = remMeta.fifoAddr;
  }

  for (int i=0; i < comm->base.vProps.ndevs; i++) {
    NCCLCHECKGOTO(wrap_ibv_reg_mr(comm->remSizesFifo.mrs+i, comm->devs[i].base.pd, &comm->remSizesFifo.elems, sizeof(int)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
  }
  comm->base.nRemDevs = remMeta.ndevs;
  for (int q = 0; q < comm->base.nqps; q++) {
    struct ncclIbQpInfo* remQpInfo   = remMeta.qpInfo + q;
    struct ncclIbDevInfo* remDevInfo = remMeta.devs + remQpInfo->devIndex;

    // Assign per-QP remDev
    comm->base.qps[q].remDevIdx = remQpInfo->devIndex;
    int devIndex = comm->base.qps[q].devIndex;
    ncclIbSendCommDev* commDev = comm->devs + devIndex;

    struct ibv_qp* qp = comm->base.qps[q].qp;
    if (remQpInfo->ece_supported && remQpInfo->ece_supported)
      NCCLCHECKGOTO(wrap_ibv_set_ece(qp, &remQpInfo->ece, &remQpInfo->ece_supported), ret, fail);

    ncclIbDev* ibDev = ncclIbDevs + commDev->base.ibDevN;
    remDevInfo->mtu = MIN(remDevInfo->mtu, ibDev->portAttr.active_mtu);
    NCCLCHECKGOTO(ncclIbRtrQp(qp, &commDev->base.gidInfo, remQpInfo->qpn, remDevInfo, false, remMeta.tc, remMeta.sl), ret, fail);
    NCCLCHECKGOTO(ncclIbRtsQp(qp), ret, fail);
  }

  if (link_layer == IBV_LINK_LAYER_ETHERNET ) { // RoCE
    for (int q = 0; q < comm->base.nqps; q++) {
      struct ncclIbQp* qp = comm->base.qps + q;
      int ibDevN = comm->devs[qp->devIndex].base.ibDevN;
      struct ncclIbDev* ibDev = ncclIbDevs + ibDevN;
      INFO(NCCL_NET,"NET/IB: IbDev %d Port %d qpn %d set_ece={supported=%d, vendor_id=0x%x, options=0x%x, comp_mask=0x%x}",
        ibDevN, ibDev->portNum, remMeta.qpInfo[q].qpn, remMeta.qpInfo[q].ece_supported, remMeta.qpInfo[q].ece.vendor_id, remMeta.qpInfo[q].ece.options, remMeta.qpInfo[q].ece.comp_mask);
    }
  }

  comm->base.nDataQps = MAX(comm->base.vProps.ndevs, comm->base.nRemDevs);

  comm->base.ready = 1;
  stage->state = ncclIbCommStateConnected;
  stage->offset = 0;

ib_send_ready:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, &comm->base.ready, sizeof(int), &stage->offset, NULL), ret, fail);
  if (stage->offset != sizeof(int)) return ncclSuccess;

  *sendComm = comm;
exit:
  if (stage->buffer) free(stage->buffer);
  stage->state = ncclIbCommStateStart;
  return ret;
fail:
  free(comm);
  goto exit;
}

NCCL_PARAM(IbWarnRailLocal, "IB_WARN_RAIL_LOCAL", 0);

ncclResult_t ncclIbCheckVProps(ncclNetVDeviceProps_t* vProps1, ncclNetVDeviceProps_t* vProps2) {
  ncclNetVDeviceProps_t  outVProps = {0};
  ncclNetVDeviceProps_t* minVProps = vProps2;
  ncclNetVDeviceProps_t* maxVProps = vProps1;
  if (vProps2->ndevs > vProps1->ndevs) {
    minVProps = vProps1;
    maxVProps = vProps2;
  }

  // Find the intersection of devices
  for (int i = 0; i < minVProps->ndevs; i++) {
    int dev = minVProps->devs[i];
    for (int j = 0; j < maxVProps->ndevs; j++) {
      // Found
      if (maxVProps->devs[j] == dev) {
        outVProps.devs[outVProps.ndevs++] = dev;
      }
    }
  }

  // In the case that at least one side has a fused NIC but there are no matching physical NICs, we should check if the user wants this
  if (ncclParamIbWarnRailLocal() && outVProps.ndevs < maxVProps->ndevs) {
    char local[128];
    int cursor = 1;
    snprintf(local, sizeof(local), "%d", vProps1->devs[0]);
    for (int i = 1; i < vProps1->ndevs; i++) {
      snprintf(local+cursor, sizeof(local)-cursor, ",%d", vProps1->devs[i]);
      cursor += 2;
    }
    char remote[128];
    snprintf(remote, sizeof(remote), "%d", vProps2->devs[0]);
    cursor = 1;
    for (int i = 1; i < vProps2->ndevs; i++) {
      snprintf(remote+cursor, sizeof(remote)-cursor, ",%d", vProps2->devs[i]);
      cursor += 2;
    }
    INFO(NCCL_NET, "NET/IB : There are mismatched physical devices between local (%s) and remote (%s). To disable this warning, set NCCL_IB_WARN_RAIL_LOCAL=0", local, remote);
  }

  return ncclSuccess;
}

NCCL_PARAM(IbGdrFlushDisable, "GDR_FLUSH_DISABLE", 0);

ncclResult_t ncclIbAccept(void* listenComm, void** recvComm,  ncclNetDeviceHandle_t** recvDevComm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbListenComm* lComm = (struct ncclIbListenComm*)listenComm;
  struct ncclIbCommStage* stage = &lComm->stage;
  struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*)stage->comm;
  int ready;
  int link_layer = IBV_LINK_LAYER_UNSPECIFIED;
  *recvComm = NULL;

  if (stage->state == ncclIbCommStateAccept)   goto ib_accept_check;
  if (stage->state == ncclIbCommStateRecvDevList) goto ib_recv_dev_list;
  if (stage->state == ncclIbCommStateSendDevList) goto ib_send_dev_list;
  if (stage->state == ncclIbCommStateRecv) goto ib_recv;
  if (stage->state == ncclIbCommStateSend) goto ib_send;
  if (stage->state == ncclIbCommStatePendingReady) goto ib_recv_ready;
  if (stage->state != ncclIbCommStateStart) {
    WARN("Listencomm in unknown state %d", stage->state);
    return ncclInternalError;
  }

  NCCLCHECK(ncclIbMalloc((void**)&rComm, sizeof(struct ncclIbRecvComm)));
  NCCLCHECKGOTO(ncclIbStatsInit(&rComm->base.stats), ret, fail);
  stage->comm = rComm;
  stage->state = ncclIbCommStateAccept;
  NCCLCHECKGOTO(ncclSocketInit(&rComm->base.sock, NULL, NCCL_SOCKET_MAGIC, ncclSocketTypeUnknown, NULL, 0, 0), ret, fail);
  NCCLCHECKGOTO(ncclSocketAccept(&rComm->base.sock, &lComm->sock), ret, fail);
  // Alloc stage->buffer here to be used for all following steps
  struct ncclIbConnectionMetadata remMeta;
  stage->offset = 0;
  NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(remMeta)));

ib_accept_check:
  NCCLCHECKGOTO(ncclSocketReady(&rComm->base.sock, &ready), ret, fail);
  if (!ready) return ncclSuccess;

  stage->state = ncclIbCommStateRecvDevList;
  stage->offset = 0;

// In the case of mismatched nDevs, we will make sure that both sides of a logical connection have the same number of RC qps
ib_recv_dev_list:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock, stage->buffer, sizeof(ncclNetVDeviceProps_t), &stage->offset, NULL));
  if (stage->offset != sizeof(ncclNetVDeviceProps_t)) return ncclSuccess;
  ncclNetVDeviceProps_t remoteVProps;
  memcpy(&remoteVProps, stage->buffer, sizeof(ncclNetVDeviceProps_t));
  if (lComm->dev >= ncclNMergedIbDevs) {
    WARN("NET/IB : Trying to use non-existent virtual device %d", lComm->dev);
    return ncclInternalError;
  }

  // Reduce the physical device list and store in the connection base
  struct ncclIbMergedDev* mergedDev;
  mergedDev = ncclIbMergedDevs + lComm->dev;
  NCCLCHECK(ncclIbCheckVProps(&mergedDev->vProps, &remoteVProps));
  rComm->base.vProps = mergedDev->vProps;
  memcpy(stage->buffer, &rComm->base.vProps, sizeof(ncclNetVDeviceProps_t));
  rComm->base.isSend = false;
  int localNqps, remoteNqps;
  localNqps  = ncclParamIbQpsPerConn() * rComm->base.vProps.ndevs; // We must have at least 1 qp per-device
  remoteNqps = ncclParamIbQpsPerConn() * remoteVProps.ndevs;
  rComm->base.nqps = remoteNqps > localNqps ? remoteNqps : localNqps; // Select max nqps (local or remote)

  stage->offset = 0;
  stage->state = ncclIbCommStateSendDevList;

ib_send_dev_list:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->base.sock, stage->buffer, sizeof(ncclNetVDeviceProps_t), &stage->offset, NULL), ret, fail);
  if (stage->offset != sizeof(ncclNetVDeviceProps_t)) return ncclSuccess;

  stage->offset = 0;
  stage->state = ncclIbCommStateRecv;

ib_recv:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock, stage->buffer, sizeof(remMeta), &stage->offset, NULL), ret, fail);
  if (stage->offset != sizeof(remMeta)) return ncclSuccess;

  /* copy back the received info */
  memcpy(&remMeta, stage->buffer, sizeof(struct ncclIbConnectionMetadata));

    // IB setup
  // Pre-declare variables because of goto
  struct ncclIbDev* ibDev;
  int ibDevN;
  struct ncclIbRecvCommDev* rCommDev;
  struct ncclIbDevInfo* remDevInfo;
  struct ncclIbQp* qp;

  mergedDev = ncclIbMergedDevs + lComm->dev;

  rComm->base.nRemDevs = remMeta.ndevs;
  if (rComm->base.nRemDevs != rComm->base.vProps.ndevs) {
    INFO(NCCL_NET, "NET/IB : Local mergedDev %s has a different number of devices=%d as remote %s %d",
      mergedDev->devName, rComm->base.vProps.ndevs, remMeta.devName, rComm->base.nRemDevs);
  }

  // Metadata to send back to requestor (sender)
  struct ncclIbConnectionMetadata meta;
  memset(&meta, 0, sizeof(meta));
  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDevN = rComm->base.vProps.devs[i];
    NCCLCHECKGOTO(ncclIbInitCommDevBase(ibDevN, &rCommDev->base, &rComm->base.stats), ret, fail);
    ibDev = ncclIbDevs + ibDevN;
    NCCLCHECKGOTO(ncclIbGetGidIndex(ibDev->context, ibDev->portNum, &ibDev->portAttr, &rCommDev->base.gidInfo.localGidIndex), ret, fail);
    NCCLCHECKGOTO(wrap_ibv_query_gid(ibDev->context, ibDev->portNum, rCommDev->base.gidInfo.localGidIndex, &rCommDev->base.gidInfo.localGid), ret, fail);
    if (link_layer == IBV_LINK_LAYER_UNSPECIFIED) link_layer = ibDev->portAttr.link_layer;
    if (link_layer != ibDev->portAttr.link_layer) {
      int ibDev0 = rComm->devs[0].base.ibDevN;
      WARN("NET/IB : Attempted to connect incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
           ibDevN, ibDev->devName, ibDev->portNum, NCCL_IB_LLSTR(ibDev->portAttr.link_layer), ibDev0, ncclIbDevs[ibDev0].devName, ncclIbDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
      return ncclInternalError;
    }
  }

  // Copy remDevInfo for things like remGidInfo, remFifoAddr, etc.
  for (int i = 0; i < remMeta.ndevs; i++) {
    rComm->base.remDevs[i] = remMeta.devs[i];
    rComm->base.remDevs[i].remoteGid.global.interface_id  = rComm->base.remDevs[i].gid.global.interface_id;
    rComm->base.remDevs[i].remoteGid.global.subnet_prefix = rComm->base.remDevs[i].gid.global.subnet_prefix;
    if (remMeta.devs[i].link_layer != link_layer) {
      int ibDev0 = rComm->devs[0].base.ibDevN;
      WARN("NET/IB : Remote %s device is incompatible with the local [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
           NCCL_IB_LLSTR(remMeta.devs[i].link_layer), ibDev0, ncclIbDevs[ibDev0].devName, ncclIbDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
      return ncclInternalError;
    }
  }

  // Stripe QP creation across merged devs
  // Make sure to get correct remote peer dev and QP info
  int remDevIndex;
  int devIndex;
  devIndex = 0;
  for (int q = 0; q < rComm->base.nqps; q++) {
    remDevIndex = remMeta.qpInfo[q].devIndex;
    remDevInfo = remMeta.devs + remDevIndex;
    qp = rComm->base.qps+q;
    rCommDev = rComm->devs + devIndex;
    qp->remDevIdx = remDevIndex;

    // Local ibDevN
    ibDevN = rComm->devs[devIndex].base.ibDevN;
    ibDev = ncclIbDevs + ibDevN;
    NCCLCHECKGOTO(ncclIbCreateQp(ibDev->portNum, &rCommDev->base, IBV_ACCESS_REMOTE_WRITE, &rComm->base.stats, qp), ret, fail);
    qp->devIndex = devIndex;
    devIndex = (devIndex + 1) % rComm->base.vProps.ndevs;

    // Set the ece (enhanced connection establishment) on this QP before RTR
    if (remMeta.qpInfo[q].ece_supported) {
      // Coverity suspects a copy-paste error below due to the use of remMeta in one argument and meta in another.
      // However, this has been confirmed to be intentional.
      // coverity[copy_paste_error]
      NCCLCHECKGOTO(wrap_ibv_set_ece(qp->qp, &remMeta.qpInfo[q].ece, &meta.qpInfo[q].ece_supported), ret, fail);


      // Query the reduced ece for this QP (matching enhancements between the requestor and the responder)
      // Store this in our own qpInfo for returning to the requestor
      if (meta.qpInfo[q].ece_supported)
        NCCLCHECKGOTO(wrap_ibv_query_ece(qp->qp, &meta.qpInfo[q].ece, &meta.qpInfo[q].ece_supported), ret, fail);
    } else {
      meta.qpInfo[q].ece_supported = 0;
    }
    NCCLCHECKGOTO(ncclIbRtrQp(qp->qp, &rCommDev->base.gidInfo, remMeta.qpInfo[q].qpn, remDevInfo, true, remMeta.tc, remMeta.sl), ret, fail);
    NCCLCHECKGOTO(ncclIbRtsQp(qp->qp), ret, fail);
  }

  rComm->flushEnabled = ((nccl_p2p_gdr_support() == ncclSuccess || nccl_p2p_dmabuf_support(lComm->dev) == ncclSuccess)
                            && (ncclParamIbGdrFlushDisable() == 0)) ? 1 : 0;

  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDev = ncclIbDevs + rCommDev->base.ibDevN;

    // Retain remote fifo info and prepare my RDMA ops
    rComm->remFifo.addr = remMeta.fifoAddr;
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->fifoMr, rCommDev->base.pd, &rComm->remFifo.elems, sizeof(struct ncclIbSendFifo)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
    rCommDev->fifoSge.lkey = rCommDev->fifoMr->lkey;
    if (ncclParamIbUseInline()) rComm->remFifo.flags = IBV_SEND_INLINE;

    // Allocate Flush dummy buffer for GPU Direct RDMA
    if (rComm->flushEnabled) {
       NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->gpuFlush.hostMr, rCommDev->base.pd, &rComm->gpuFlushHostMem, sizeof(int), IBV_ACCESS_LOCAL_WRITE), ret, fail);
      rCommDev->gpuFlush.sge.addr = (uint64_t)&rComm->gpuFlushHostMem;
      rCommDev->gpuFlush.sge.length = 1;
      rCommDev->gpuFlush.sge.lkey = rCommDev->gpuFlush.hostMr->lkey;
      NCCLCHECKGOTO(ncclIbCreateQp(ibDev->portNum, &rCommDev->base, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ, &rComm->base.stats, &rCommDev->gpuFlush.qp), ret, fail);;
      struct ncclIbDevInfo devInfo;
      devInfo.lid         = ibDev->portAttr.lid;
      devInfo.link_layer  = ibDev->portAttr.link_layer;
      devInfo.ib_port     = ibDev->portNum;
      devInfo.gid.global.subnet_prefix        = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
      devInfo.gid.global.interface_id         = rCommDev->base.gidInfo.localGid.global.interface_id;
      devInfo.is_global   = (ncclParamIbIsGlobal()
#if HAVE_DECL_IBV_QPF_GRH_REQUIRED
                     || (ibDev->portAttr.flags & IBV_QPF_GRH_REQUIRED)
#endif
      );
      devInfo.mtu         = ibDev->portAttr.active_mtu;
      NCCLCHECKGOTO(ncclIbRtrQp(rCommDev->gpuFlush.qp.qp, &rCommDev->base.gidInfo, rCommDev->gpuFlush.qp.qp->qp_num, &devInfo, false, remMeta.tc, remMeta.sl), ret, fail);
      NCCLCHECKGOTO(ncclIbRtsQp(rCommDev->gpuFlush.qp.qp), ret, fail);
    }

    // Fill Handle
    meta.devs[i].lid                            = ibDev->portAttr.lid;
    meta.devs[i].link_layer                     = rCommDev->base.gidInfo.link_layer = ibDev->portAttr.link_layer;
    meta.devs[i].ib_port                        = ibDev->portNum;
    meta.devs[i].gid.global.subnet_prefix       = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
    meta.devs[i].gid.global.interface_id        = rCommDev->base.gidInfo.localGid.global.interface_id;
    meta.devs[i].is_global  = (ncclParamIbIsGlobal()
#if HAVE_DECL_IBV_QPF_GRH_REQUIRED
                     || (ibDev->portAttr.flags & IBV_QPF_GRH_REQUIRED)
#endif
      );

    meta.devs[i].mtu                            = ibDev->portAttr.active_mtu;


    // Prepare sizes fifo
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&rComm->devs[i].sizesFifoMr, rComm->devs[i].base.pd, rComm->sizesFifo, sizeof(int)*MAX_REQUESTS*NCCL_NET_IB_MAX_RECVS, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
    meta.devs[i].fifoRkey = rComm->devs[i].sizesFifoMr->rkey;
  }
  meta.fifoAddr = (uint64_t)rComm->sizesFifo;
  meta.sl = remMeta.sl;
  meta.tc = remMeta.tc;

  for (int q = 0; q < rComm->base.nqps; q++) {
    meta.qpInfo[q].qpn      = rComm->base.qps[q].qp->qp_num;
    meta.qpInfo[q].devIndex = rComm->base.qps[q].devIndex;
  }

  meta.ndevs = rComm->base.vProps.ndevs;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);
  rComm->base.nDataQps = MAX(rComm->base.vProps.ndevs, rComm->base.nRemDevs);

  stage->state = ncclIbCommStateSend;
  stage->offset = 0;
  if (stage->buffer) {
    free(stage->buffer);
    stage->buffer = NULL;
  }
  NCCLCHECKGOTO(ncclIbMalloc((void**)&stage->buffer, sizeof(struct ncclIbConnectionMetadata)), ret, fail);
  memcpy(stage->buffer, &meta, sizeof(struct ncclIbConnectionMetadata));
ib_send:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->base.sock, stage->buffer, sizeof(struct ncclIbConnectionMetadata), &stage->offset, NULL), ret, fail);
  if (stage->offset < sizeof(struct ncclIbConnectionMetadata)) return ncclSuccess;

  stage->offset = 0;
  stage->state = ncclIbCommStatePendingReady;

ib_recv_ready:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV,  &rComm->base.sock, &rComm->base.ready, sizeof(int), &stage->offset, NULL), ret, fail);
  if (stage->offset != sizeof(int)) return ncclSuccess;

  *recvComm = rComm;
exit:
  /* reset lComm stage */
  if (stage->buffer) free(stage->buffer);
  stage->state = ncclIbCommStateStart;
  stage->offset = 0;
  stage->comm = NULL;
  stage->buffer = NULL;
  return ret;
fail:
  free(rComm);
  goto exit;
}

ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req) {
  for (int i=0; i<MAX_REQUESTS; i++) {
    struct ncclIbRequest* r = base->reqs+i;
    if (r->type == NCCL_NET_IB_REQ_UNUSED) {
      r->base = base;
      r->sock = NULL;
      r->devBases[0] = NULL;
      r->devBases[1] = NULL;
      r->events[0] = r->events[1] = 0;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/IB : unable to allocate requests");
  *req = NULL;
  return ncclInternalError;
}

ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* r) {
  r->type = NCCL_NET_IB_REQ_UNUSED;
  return ncclSuccess;
}

ncclResult_t ncclIbTest(void* request, int* done, int* size);

ncclResult_t ncclIbRegMrDmaBufInternal(ncclIbNetCommDevBase* base, void* data, size_t size, int type, uint64_t offset, int fd, struct ibv_mr** mhandle) {
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);

  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize-1)/pageSize;
  ncclResult_t res;
  pthread_mutex_lock(&ncclIbDevs[base->ibDevN].lock);
  for (int slot=0; /*true*/; slot++) {
    if (slot == cache->population || addr < cache->slots[slot].addr) { // didn't find in cache
      if (cache->population == cache->capacity) { // must grow cache
        cache->capacity = cache->capacity < 32 ? 32 : 2*cache->capacity;
        NCCLCHECKGOTO(ncclRealloc((void **)&cache->slots, sizeof(struct ncclIbMr)*cache->population, sizeof(struct ncclIbMr)*cache->capacity), res, returning);
      }
      // Deregister / register
      struct ibv_mr* mr;
      unsigned int flags = IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
      if (ncclIbRelaxedOrderingEnabled) flags |= IBV_ACCESS_RELAXED_ORDERING;
      if (fd != -1) {
        /* DMA-BUF support */
        NCCLCHECKGOTO(wrap_ibv_reg_dmabuf_mr(&mr, base->pd, offset, pages*pageSize, addr, fd, flags), res, returning);
      } else {
        if (ncclIbRelaxedOrderingEnabled) {
          // Use IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
          NCCLCHECKGOTO(wrap_ibv_reg_mr_iova2(&mr, base->pd, (void*)addr, pages*pageSize, addr, flags), res, returning);
        }
        else {
          NCCLCHECKGOTO(wrap_ibv_reg_mr(&mr, base->pd, (void*)addr, pages*pageSize, flags), res, returning);
        }
      }
      TRACE(NCCL_INIT|NCCL_NET,"regAddr=0x%lx size=%lld rkey=0x%x lkey=0x%x fd=%d", (unsigned long)addr, (long long)pages*pageSize, mr->rkey, mr->lkey, fd);
      if (slot != cache->population) memmove(cache->slots+slot+1, cache->slots+slot, (cache->population-slot)*sizeof(struct ncclIbMr));
      cache->slots[slot].addr = addr;
      cache->slots[slot].pages = pages;
      cache->slots[slot].refs = 1;
      cache->slots[slot].mr = mr;
      cache->population += 1;
      *mhandle = (void*)mr;
      res = ncclSuccess;
      goto returning;
     } else if ((addr >= cache->slots[slot].addr) &&
        ((addr-cache->slots[slot].addr)/pageSize+pages) <= cache->slots[slot].pages) {
      cache->slots[slot].refs += 1;
      *mhandle = cache->slots[slot].mr;
      res = ncclSuccess;
      goto returning;
    }
  }
returning:
  pthread_mutex_unlock(&ncclIbDevs[base->ibDevN].lock);
  return res;
}

struct ncclIbNetCommDevBase* ncclIbGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex) {
  if (base->isSend) {
    struct ncclIbSendComm* sComm = (struct ncclIbSendComm*) base;
    return &sComm->devs[devIndex].base;
  } else {
    struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*) base;
    return &rComm->devs[devIndex].base;
  }
}

/* DMA-BUF support */
ncclResult_t ncclIbRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
  ncclResult_t ret = ncclSuccess;
  assert(size > 0);
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*) comm;
  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) malloc(sizeof(struct ncclIbMrHandle));
  for (int i = 0; i < base->vProps.ndevs; i++) {
    // Each ncclIbNetCommDevBase is at different offset in send and recv netComms
    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
    NCCLCHECKGOTO(ncclIbRegMrDmaBufInternal(devComm, data, size, type, offset, fd, mhandleWrapper->mrs + i), ret, fail);
  }
  *mhandle = (void*) mhandleWrapper;
exit:
  return ret;
fail:
  free(mhandleWrapper);
  goto exit;
}

ncclResult_t ncclIbRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  return ncclIbRegMrDmaBuf(comm, data, size, type, 0ULL, -1, mhandle);
}

ncclResult_t ncclIbDeregMrInternal(ncclIbNetCommDevBase* base, struct ibv_mr* mhandle) {
  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
  ncclResult_t res;
  pthread_mutex_lock(&ncclIbDevs[base->ibDevN].lock);
  for (int i=0; i < cache->population; i++) {
    if (mhandle == cache->slots[i].mr) {
      if (0 == --cache->slots[i].refs) {
        memmove(&cache->slots[i], &cache->slots[--cache->population], sizeof(struct ncclIbMr));
        if (cache->population == 0) {
          free(cache->slots);
          cache->slots = NULL;
          cache->capacity = 0;
        }
         NCCLCHECKGOTO(wrap_ibv_dereg_mr(mhandle), res, returning);
      }
      res = ncclSuccess;
      goto returning;
    }
  }
  WARN("NET/IB: could not find mr %p inside cache of %d entries", mhandle, cache->population);
  res = ncclInternalError;
returning:
  pthread_mutex_unlock(&ncclIbDevs[base->ibDevN].lock);
  return res;
}

ncclResult_t ncclIbDeregMr(void* comm, void* mhandle) {
  if (mhandle == NULL) return ncclSuccess;

  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandle;
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*) comm;
  for (int i = 0; i < base->vProps.ndevs; i++) {
    // Each ncclIbNetCommDevBase is at different offset in send and recv netComms
    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
    NCCLCHECK(ncclIbDeregMrInternal(devComm, mhandleWrapper->mrs[i]));
  }
  free(mhandleWrapper);
  return ncclSuccess;
}

NCCL_PARAM(IbSplitDataOnQps, "IB_SPLIT_DATA_ON_QPS", 0);

ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot,  void* pHandle) {
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  volatile struct ncclIbSendFifo* slots = comm->fifo[slot];
  int nreqs = slots[0].nreqs;
  if (nreqs > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;

  uint64_t wr_id = 0ULL;
  for (int r=0; r<nreqs; r++) {
    struct ibv_send_wr* wr = comm->wrs+r;
    memset(wr, 0, sizeof(struct ibv_send_wr));

    struct ibv_sge* sge = comm->sges+r;
    sge->addr=(uintptr_t)reqs[r]->send.data;

    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->send_flags = 0;
    wr->wr.rdma.remote_addr = slots[r].addr;
    wr->next = wr + 1;
    wr_id += (reqs[r] - comm->base.reqs) << (r*8);
  }

  // Write size as immediate data. In the case of multi-send, only write
  // 0 or 1 as size to indicate whether there was data sent or received.
  uint32_t immData = 0;
  if (nreqs == 1) {
    immData = reqs[0]->send.size;
  } else {
    int* sizes = comm->remSizesFifo.elems[slot];
    for (int r=0; r<nreqs; r++) sizes[r] = reqs[r]->send.size;
    comm->remSizesFifo.sge.addr = (uint64_t)sizes;
    comm->remSizesFifo.sge.length = nreqs*sizeof(int);
  }

  struct ibv_send_wr* lastWr = comm->wrs+nreqs-1;
  if (nreqs > 1 || (comm->ar && reqs[0]->send.size > ncclParamIbArThreshold())) {
    // When using ADAPTIVE_ROUTING, send the bulk of the data first as an
    // RDMA_WRITE, then a 0-byte RDMA_WRITE_WITH_IMM to trigger a remote
    // completion.
    lastWr++;
    memset(lastWr, 0, sizeof(struct ibv_send_wr));
    if (nreqs > 1) {
      // Write remote sizes Fifo
      lastWr->wr.rdma.remote_addr = comm->remSizesFifo.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(int);
      lastWr->num_sge = 1;
      lastWr->sg_list = &comm->remSizesFifo.sge;
    }
  }
  lastWr->wr_id = wr_id;
  lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  lastWr->imm_data = immData;
  lastWr->next = NULL;
  lastWr->send_flags = IBV_SEND_SIGNALED;

  // Multi-QP: make sure IB writes are multiples of 128B so that LL and LL128 protocols still work
  const int align = 128;
  int nqps = ncclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.nDataQps;
  for (int i = 0; i < nqps; i++) {
    int qpIndex = comm->base.qpIndex;
    ncclIbQp* qp = comm->base.qps + qpIndex;
    int devIndex = qp->devIndex;
    for (int r=0; r<nreqs; r++) {
      // Track this event for completion
      //ncclIbAddEvent(reqs[r], devIndex, &comm->devs[devIndex].base);

      // Select proper rkey (needed even for 0-size send)
      comm->wrs[r].wr.rdma.rkey = slots[r].rkeys[qp->remDevIdx];

      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      int length = MIN(reqs[r]->send.size-reqs[r]->send.offset, chunkSize);
      if (length <= 0) {
        comm->wrs[r].sg_list = NULL;
        comm->wrs[r].num_sge = 0;
      } else {
        // Select proper lkey
        comm->sges[r].lkey = reqs[r]->send.lkeys[devIndex];
        comm->sges[r].length = length;
        comm->wrs[r].sg_list = comm->sges+r;
        comm->wrs[r].num_sge = 1;
      }
    }

    if (nreqs > 1) {
      // Also make sure lastWr writes remote sizes using the right lkey
      comm->remSizesFifo.sge.lkey = comm->remSizesFifo.mrs[devIndex]->lkey;
      lastWr->wr.rdma.rkey = comm->remSizesFifo.rkeys[devIndex];
    }

    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(qp->qp, comm->wrs, &bad_wr));

    for (int r=0; r<nreqs; r++) {
      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      reqs[r]->send.offset += chunkSize;
      comm->sges[r].addr += chunkSize;
      comm->wrs[r].wr.rdma.remote_addr += chunkSize;
    }

    // Select the next qpIndex
    comm->base.qpIndex = (comm->base.qpIndex+1) % comm->base.nqps;
  }

  return ncclSuccess;
}

ncclResult_t ncclIbIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm->base.ready == 0) { WARN("NET/IB: ncclIbIsend() called when comm->base.ready == 0"); return ncclInternalError; }
  if (comm->base.ready == 0) { *request = NULL; return ncclSuccess; }
  NCCLCHECK(ncclIbStatsCheckFatalCount(&comm->base.stats,__func__));

  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandle;

  // Wait for the receiver to have posted the corresponding receive
  int nreqs = 0;
  volatile struct ncclIbSendFifo* slots;

  int slot = (comm->fifoHead) % MAX_REQUESTS;
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  slots = comm->fifo[slot];
  uint64_t idx = comm->fifoHead+1;
  if (slots[0].idx != idx) { *request = NULL; return ncclSuccess; }
  nreqs = slots[0].nreqs;
  // Wait until all data has arrived
  for (int r=1; r<nreqs; r++) while(slots[r].idx != idx);
  __sync_synchronize(); // order the nreqsPtr load against tag/rkey/addr loads below
  for (int r=0; r<nreqs; r++) {
    if (reqs[r] != NULL || slots[r].tag != tag) continue;

    if (size > slots[r].size) size = slots[r].size;
    // Sanity checks
    if (slots[r].size < 0 || slots[r].addr == 0 || slots[r].rkeys[0] == 0) {
      char line[SOCKET_NAME_MAXLEN + 1];
      union ncclSocketAddress addr;
      ncclSocketGetAddr(&comm->base.sock, &addr);
      WARN("NET/IB : req %d/%d tag %x peer %s posted incorrect receive info: size %ld addr %lx rkeys[0]=%x",
        r, nreqs, tag, ncclSocketToString(&addr, line, 1), slots[r].size, slots[r].addr, slots[r].rkeys[0]);
    }
    struct ncclIbRequest* req;
    NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
    req->type = NCCL_NET_IB_REQ_SEND;
    req->sock = &comm->base.sock;
    req->base = &comm->base;
    req->nreqs = nreqs;
    req->send.size = size;
    req->send.data = data;
    req->send.offset = 0;

    // Populate events
    int nEvents = ncclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.nDataQps;
    int qpIndex = comm->base.qpIndex;
    // Count down
    while (nEvents > 0) {
      ncclIbQp* qp = comm->base.qps + qpIndex;
      int devIndex = qp->devIndex;
      ncclIbAddEvent(req, devIndex, &comm->devs[devIndex].base);
      // Track the valid lkey for this RDMA_Write
      req->send.lkeys[devIndex] = mhandleWrapper->mrs[devIndex]->lkey;
      nEvents--;
      // Don't update comm->base.qpIndex yet, we need to run through this same set of QPs inside ncclIbMultiSend()
      qpIndex = (qpIndex+1)%comm->base.nqps;
    }

    // Store all lkeys
    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      req->send.lkeys[i] = mhandleWrapper->mrs[i]->lkey;
    }

    *request = reqs[r] = req;

    // If this is a multi-recv, send only when all requests have matched.
    for (int r=0; r<nreqs; r++) {
      if (reqs[r] == NULL) return ncclSuccess;
    }

    TIME_START(0);
    NCCLCHECK(ncclIbMultiSend(comm, slot, phandle));

    // Clear slots[0]->nreqs, as well as other fields to help debugging and sanity checks
    memset((void*)slots, 0, sizeof(struct ncclIbSendFifo));
    memset(reqs, 0, NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbRequest*));
    comm->fifoHead++;
    TIME_STOP(0);
    return ncclSuccess;
  }

  *request = NULL;
  return ncclSuccess;
}

ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, int n, void** data, size_t* sizes, int* tags, void** mhandles, struct ncclIbRequest* req) {
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));

  int slot = comm->remFifo.fifoTail%MAX_REQUESTS;
  req->recv.sizes = comm->sizesFifo[slot];
  for (int i=0; i<n; i++) req->recv.sizes[i] = 0;
  struct ncclIbSendFifo* localElem = comm->remFifo.elems[slot];

  // Select the next devIndex (local) and QP to use for posting this CTS message
  // Since QPs are initialized by striping across devIndex, we can simply assign this to the same value
  ncclIbQp* ctsQp = comm->base.qps + comm->base.devIndex;
  comm->base.devIndex = (comm->base.devIndex + 1) % comm->base.vProps.ndevs;

  for (int i=0; i<n; i++) {
    localElem[i].addr = (uint64_t)data[i];
    struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandles[i];

    // Send all applicable rkeys
   for (int j = 0; j < comm->base.vProps.ndevs; j++)
      localElem[i].rkeys[j] = mhandleWrapper->mrs[j]->rkey;

    localElem[i].nreqs = n;
    localElem[i].size = sizes[i]; // Sanity/Debugging
    localElem[i].tag = tags[i];
    localElem[i].idx = comm->remFifo.fifoTail+1;
  }

  wr.wr.rdma.remote_addr = comm->remFifo.addr + slot*NCCL_NET_IB_MAX_RECVS*sizeof(struct ncclIbSendFifo);

  // Lookup the correct fifoRkey
  wr.wr.rdma.rkey = comm->base.remDevs[ctsQp->remDevIdx].fifoRkey;

  // Set the correct sge properties
  comm->devs[ctsQp->devIndex].fifoSge.addr   = (uint64_t)localElem;
  comm->devs[ctsQp->devIndex].fifoSge.length = n*sizeof(struct ncclIbSendFifo);
  wr.sg_list = &comm->devs[ctsQp->devIndex].fifoSge;

  wr.num_sge = 1;

  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = comm->remFifo.flags; // IBV_SEND_INLINE

  // We need to occasionally post a request with the IBV_SEND_SIGNALED flag, otherwise
  // the send queue will never empty.
  //
  // From https://www.rdmamojo.com/2014/06/30/working-unsignaled-completions/
  // "How to use Unsignaled Completion?" / "Gotchas and Pitfalls"
  // All posted Send Requested, Signaled and Unsignaled, are considered outstanding until
  // a Work Completion that they, or Send Requests that were posted after them, was polled
  // from the Completion Queue associated with the Send Queue. This means if one works with
  // a Queue Pair that was configured to work with Unsignaled Completions, he must make
  // sure that occasionally (before the Send Queue is full with outstanding Send Requests)
  // a Send Request that generate Work Completion will be posted.
  //
  // Not following this rule may lead to a case that the Send Queue is full with Send
  // Requests that won't generate Work Completion:
  //
  //  - The Send Queue is full, so no new Send Requests can be posted to it
  //  - The Send Queue can't be emptied, since no Work Completion can be generated anymore
  //    (the reason is that no Work Completion, that can generate Work Completion that
  //    polling it will empty the Send Queue, can be posted)
  //  - The status of all posted Send Request is considered unknown
  //
  // slot == devIndex - When writing to fifo slot N, and this QP lives on device index N, it should send signalled.
  // This works out that each fifo posting QP gets drained
  if (slot == ctsQp->devIndex) {

    wr.send_flags |= IBV_SEND_SIGNALED;
    wr.wr_id = req - comm->base.reqs;
    ncclIbAddEvent(req, ctsQp->devIndex, &comm->devs[ctsQp->devIndex].base);
  }

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(ctsQp->qp, &wr, &bad_wr));
  comm->remFifo.fifoTail++;

  return ncclSuccess;
}

ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm->base.ready == 0) { WARN("NET/IB: ncclIbIrecv() called when comm->base.ready == 0"); return ncclInternalError; }
  if (comm->base.ready == 0) { *request = NULL; return ncclSuccess; }
  NCCLCHECK(ncclIbStatsCheckFatalCount(&comm->base.stats,__func__));

  if (n > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_RECV;
  req->sock = &comm->base.sock;
  req->nreqs = n;

  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));

  wr.wr_id = req - comm->base.reqs;
  wr.sg_list = NULL;
  wr.num_sge = 0;

  TIME_START(1);
  // Select either all QPs, or one qp per-device
  const int nqps = ncclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.nDataQps;

  // Post recvs
  struct ibv_recv_wr* bad_wr;
  for (int i = 0; i < nqps; i++) {
    struct ncclIbQp* qp = comm->base.qps + comm->base.qpIndex;
    ncclIbAddEvent(req, qp->devIndex, &comm->devs[qp->devIndex].base);
    NCCLCHECK(wrap_ibv_post_recv(qp->qp, &wr, &bad_wr));
    comm->base.qpIndex = (comm->base.qpIndex+1)%comm->base.nqps;
  }
  TIME_STOP(1);

  // Post to FIFO to notify sender
  TIME_START(2);
  NCCLCHECK(ncclIbPostFifo(comm, n, data, sizes, tags, mhandles, req));
  TIME_STOP(2);

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  int last = -1;
  for (int i=0; i<n; i++) if (sizes[i]) last = i;
  if (comm->flushEnabled == 0 || last == -1) return ncclSuccess;

  // Only flush once using the last non-zero receive
  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  struct ncclIbMrHandle* mhandle = (struct ncclIbMrHandle*) mhandles[last];

  // We don't know which devIndex the recv was on, so we flush on all devices
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = req - comm->base.reqs;

    wr.wr.rdma.remote_addr = (uint64_t)data[last];
    wr.wr.rdma.rkey = mhandle->mrs[i]->rkey;
    wr.sg_list = &comm->devs[i].gpuFlush.sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;

    TIME_START(4);
    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(comm->devs[i].gpuFlush.qp.qp, &wr, &bad_wr));
    TIME_STOP(4);
    ncclIbAddEvent(req, i, &comm->devs[i].base);
  }

  *request = req;
  return ncclSuccess;
}

#define HCA_NAME(req, index) ((req)->devBases[(index)]->pd->context->device->name)

ncclResult_t ncclIbTest(void* request, int* done, int* sizes) {
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;
  *done = 0;

  while (1) {
    NCCLCHECK(ncclIbStatsCheckFatalCount(&r->base->stats,__func__));
    if (r->events[0] == 0 && r->events[1] == 0 && r->events[2] == 0 && r->events[3] == 0) {
      TRACE(NCCL_NET, "r=%p done", r);
      *done = 1;
      if (sizes && r->type == NCCL_NET_IB_REQ_RECV) {
        for (int i=0; i<r->nreqs; i++) sizes[i] = r->recv.sizes[i];
      }
      if (sizes && r->type == NCCL_NET_IB_REQ_SEND) {
        sizes[0] = r->send.size;
      }

      if (sizes && r->type == NCCL_NET_IB_REQ_SEND) {
        sizes[0] = r->send.size;
      }
     
      // Stop all remaining Qp events for this event
      NCCLCHECK(ncclIbFreeRequest(r));
      return ncclSuccess;
    }

    int totalWrDone = 0;
    int wrDone = 0;
    struct ibv_wc wcs[4];
        for (int i = 0; i < NCCL_IB_MAX_DEVS_PER_NIC; i++) {
      TIME_START(3);
      // If we expect any completions from this device's CQ
      if (r->events[i]) {
        NCCLCHECK(wrap_ibv_poll_cq(r->devBases[i]->cq, 4, wcs, &wrDone));
        totalWrDone += wrDone;
        if (wrDone == 0) { TIME_CANCEL(3); } else { TIME_STOP(3); }
        if (wrDone == 0) continue;
        for (int w=0; w<wrDone; w++) {
          struct ibv_wc *wc = wcs+w;
          if (wc->status != IBV_WC_SUCCESS) {
            union ncclSocketAddress addr;
            ncclSocketGetAddr(r->sock, &addr);
            char localGidString[INET6_ADDRSTRLEN] = "";
            char remoteGidString[INET6_ADDRSTRLEN] = "";
            const char* localGidStr = NULL, *remoteGidStr = NULL;
            if (r->devBases[i]->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
              localGidStr = ibvGetGidStr(&r->devBases[i]->gidInfo.localGid, localGidString, sizeof(localGidString));
              remoteGidStr = ibvGetGidStr(&r->base->remDevs[i].remoteGid, remoteGidString, sizeof(remoteGidString));
            }

            char line[SOCKET_NAME_MAXLEN+1];
	          char *hcaName = r->devBases[i]->pd->context->device->name;
	          WARN("NET/IB: Got completion from peer %s with status=%d opcode=%d len=%u vendor err %u (%s)%s%s%s%s hca %s",
                ncclSocketToString(&addr, line, 1), wc->status, wc->opcode, wc->byte_len, wc->vendor_err, reqTypeStr[r->type],
                localGidStr ?  " localGid ":"", localGidString, remoteGidStr ? " remoteGids":"", remoteGidString, hcaName);
            return ncclRemoteError;
          }

          union ncclSocketAddress addr;
          ncclSocketGetAddr(r->sock, &addr);
          struct ncclIbRequest* req = r->base->reqs+(wc->wr_id & 0xff);

          #ifdef ENABLE_TRACE
          char line[SOCKET_NAME_MAXLEN+1];
          TTRACE(NCCL_NET, "Got completion from peer %s with status=%d opcode=%d len=%u wr_id=%lu r=%p type=%d events={%d,%d}, i=%d",
              ncclSocketToString(&addr, line, 1), wc->status, wc->opcode,wc->byte_len, wc->wr_id, req, req->type, req->events[0], req->events[1], i);
          #endif
          if (req && req->type == NCCL_NET_IB_REQ_SEND) {
            for (int j = 0; j < req->nreqs; j++) {
              struct ncclIbRequest* sendReq = r->base->reqs+((wc->wr_id >> (j*8)) & 0xff);
              if ((sendReq->events[i] <= 0)) {
                WARN("NET/IB: sendReq(%p)->events={%d,%d}, i=%d, j=%d <= 0", sendReq, sendReq->events[0], sendReq->events[1], i, j);
                return ncclInternalError;
              }
              sendReq->events[i]--;
            }
          } else {
            if (req && wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
              if (req->type != NCCL_NET_IB_REQ_RECV) {
                WARN("NET/IB: wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM and req->type=%d", req->type);
                return ncclInternalError;
              }
              if (req->nreqs == 1) {
                req->recv.sizes[0] = wc->imm_data;
              }
            }
            req->events[i]--;
          }
        }
        // Once the IB fatal event is reported in the async thread, we want to propagate this error
        // to communicator and prevent further polling to reduce error pollution.
        NCCLCHECK(ncclIbStatsCheckFatalCount(&ncclIbDevs[r->devBases[i]->ibDevN].stats,__func__));
      }
    }

    // If no CQEs found on any device, return and come back later
    if (totalWrDone == 0) return ncclSuccess;
  }
}

ncclResult_t ncclIbCloseSend(void* sendComm) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->base.sock, false));

    for (int q = 0; q < comm->base.nqps; q++)
      if (comm->base.qps[q].qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q].qp));

    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      struct ncclIbSendCommDev* commDev = comm->devs + i;
      if (commDev->fifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->fifoMr));
      if (comm->remSizesFifo.mrs[i] != NULL) NCCLCHECK(wrap_ibv_dereg_mr(comm->remSizesFifo.mrs[i]));
      NCCLCHECK(ncclIbDestroyBase(&commDev->base));
    }
    free(comm);
  }
  TIME_PRINT("IB");
  return ncclSuccess;
}

ncclResult_t ncclIbCloseRecv(void* recvComm) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->base.sock, false));

    for (int q = 0; q < comm->base.nqps; q++)
      if (comm->base.qps[q].qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q].qp));

    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      struct ncclIbRecvCommDev* commDev = comm->devs + i;
      if (comm->flushEnabled) {
        if (commDev->gpuFlush.qp.qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(commDev->gpuFlush.qp.qp));
        if (commDev->gpuFlush.hostMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->gpuFlush.hostMr));
      }
      if (commDev->fifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->fifoMr));
      if (commDev->sizesFifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->sizesFifoMr));
      NCCLCHECK(ncclIbDestroyBase(&commDev->base));
    }
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclIbCloseListen(void* listenComm) {
  struct ncclIbListenComm* comm = (struct ncclIbListenComm*)listenComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->sock, false));
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclIbMakeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  pthread_mutex_lock(&ncclIbLock);
  ncclResult_t res = ncclIbMakeVDeviceInternal(d, props, ncclNIbDevs, &ncclNMergedIbDevs);
  pthread_mutex_unlock(&ncclIbLock);
  return res;
}

ncclResult_t ncclIbConnect_v9(int dev, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm) {
  return ncclIbConnect(dev, NULL, opaqueHandle, sendComm, sendDevComm);
}

ncclResult_t ncclIbInit_v9(ncclDebugLogger_t logFunction) {
  return ncclIbInit(logFunction, NULL);
}

ncclResult_t ncclIbIsend_v9(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request) {
  return ncclIbIsend(sendComm, data, size, tag, mhandle, NULL, request);
}

ncclResult_t ncclIbIrecv_v9(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request) {
  return ncclIbIrecv(recvComm, n, data, sizes, tags, mhandles, NULL, request);
}

ncclResult_t ncclIbIsend_v8(void* sendComm, void* data, int size, int tag, void* mhandle, void** request)
{
  return ncclIbIsend_v9(sendComm, data, (size_t)size, tag, mhandle, request);
}

ncclResult_t ncclIbIrecv_v8(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request)
{
  size_t sizesOut[NCCL_NET_IB_MAX_RECVS];
  for (int i=0; i<n; i++) sizesOut[i] = sizes[i];
  return ncclIbIrecv_v9(recvComm, 1, data, sizesOut, tags, mhandles, request);
}

ncclResult_t ncclIbRegMr_v7(void* comm, void* data, int size, int type, void** mhandle) {
  return ncclIbRegMrDmaBuf(comm, data, (size_t)size, type, 0ULL, -1, mhandle);
}

ncclResult_t ncclIbConnect_v6(int dev, void* opaqueHandle, void** sendComm) {
  ncclNetDeviceHandle_v7_t* handle = NULL;
  return ncclIbConnect_v9(dev, opaqueHandle, sendComm, &handle);
}

ncclResult_t ncclIbAccept_v6(void* listenComm, void** recvComm) {
  ncclNetDeviceHandle_v7_t* handle = NULL;
  return ncclIbAccept(listenComm, recvComm, &handle);
}


const ncclNet_v10_t ibPlugin_v10 = {
  .name = "IBext_v10",
  .init = ncclIbInit,
  .devices = ncclIbDevices,
  .getProperties = ncclIbGetProperties,
  .listen = ncclIbListen,
  .connect = ncclIbConnect,
  .accept = ncclIbAccept,
  .regMr = ncclIbRegMr,
  .regMrDmaBuf = ncclIbRegMrDmaBuf,
  .deregMr = ncclIbDeregMr,
  .isend = ncclIbIsend,
  .irecv = ncclIbIrecv,
  .iflush = ncclIbIflush,
  .test = ncclIbTest,
  .closeSend = ncclIbCloseSend,
  .closeRecv = ncclIbCloseRecv,
  .closeListen = ncclIbCloseListen,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */,
  ncclIbMakeVDevice
};

const ncclNet_v9_t ibPlugin_v9 = {
  .name = "IBext_v9",
  .init = ncclIbInit_v9,
  .devices = ncclIbDevices,
  .getProperties = ncclIbGetProperties_v9,
  .listen = ncclIbListen,
  .connect = ncclIbConnect_v9,
  .accept = ncclIbAccept,
  .regMr = ncclIbRegMr,
  .regMrDmaBuf = ncclIbRegMrDmaBuf,
  .deregMr = ncclIbDeregMr,
  .isend = ncclIbIsend_v9,
  .irecv = ncclIbIrecv_v9,
  .iflush = ncclIbIflush,
  .test = ncclIbTest,
  .closeSend = ncclIbCloseSend,
  .closeRecv = ncclIbCloseRecv,
  .closeListen = ncclIbCloseListen,
  .getDeviceMr = NULL,
  .irecvConsumed = NULL,
  .makeVDevice = ncclIbMakeVDevice,
};


const ncclNet_v8_t ibPlugin_v8 = {
  .name = "IBext_v8",
  .init = ncclIbInit_v9,
  .devices = ncclIbDevices,
  .getProperties = ncclIbGetProperties_v8,
  .listen = ncclIbListen,
  .connect = ncclIbConnect_v9,
  .accept = ncclIbAccept,
  .regMr = ncclIbRegMr,
  .regMrDmaBuf = ncclIbRegMrDmaBuf,
  .deregMr = ncclIbDeregMr,
  .isend = ncclIbIsend_v8,
  .irecv = ncclIbIrecv_v8,
  .iflush = ncclIbIflush,
  .test = ncclIbTest,
  .closeSend = ncclIbCloseSend,
  .closeRecv = ncclIbCloseRecv,
  .closeListen = ncclIbCloseListen,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */
};

const ncclNet_v7_t ibPlugin_v7 = {
  .name = "IBext_v7",
  .init = ncclIbInit_v9,
  .devices = ncclIbDevices,
  .getProperties = ncclIbGetProperties_v7,
  .listen = ncclIbListen,
  .connect = ncclIbConnect_v9,
  .accept = ncclIbAccept,
  .regMr = ncclIbRegMr_v7,
  .regMrDmaBuf = ncclIbRegMrDmaBuf,
  .deregMr = ncclIbDeregMr,
  .isend = ncclIbIsend_v8,
  .irecv = ncclIbIrecv_v8,
  .iflush = ncclIbIflush,
  .test = ncclIbTest,
  .closeSend = ncclIbCloseSend,
  .closeRecv = ncclIbCloseRecv,
  .closeListen = ncclIbCloseListen,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */
};

const ncclNet_v6_t ibPlugin_v6 = {
  .name = "IBext_v6",
  .init = ncclIbInit_v9,
  .devices = ncclIbDevices,
  .getProperties = ncclIbGetProperties_v6,
  .listen = ncclIbListen,
  .connect = ncclIbConnect_v6,
  .accept = ncclIbAccept_v6,
  .regMr = ncclIbRegMr_v7,
  .regMrDmaBuf = ncclIbRegMrDmaBuf,
  .deregMr = ncclIbDeregMr,
  .isend = ncclIbIsend_v8,
  .irecv = ncclIbIrecv_v8,
  .iflush = ncclIbIflush,
  .test = ncclIbTest,
  .closeSend = ncclIbCloseSend,
  .closeRecv = ncclIbCloseRecv,
  .closeListen = ncclIbCloseListen,
};

const ncclNet_v5_t ibPlugin_v5 = {
  .name = "IBext_v5",
  .init = ncclIbInit_v9,
  .devices = ncclIbDevices,
  .getProperties = ncclIbGetProperties_v6,
  .listen = ncclIbListen,
  .connect = ncclIbConnect_v6,
  .accept = ncclIbAccept_v6,
  .regMr = ncclIbRegMr_v7,
  .deregMr = ncclIbDeregMr,
  .isend = ncclIbIsend_v8,
  .irecv = ncclIbIrecv_v8,
  .iflush = ncclIbIflush,
  .test = ncclIbTest,
  .closeSend = ncclIbCloseSend,
  .closeRecv = ncclIbCloseRecv,
  .closeListen = ncclIbCloseListen,
};
