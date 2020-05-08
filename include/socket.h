/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SOCKET_H_
#define NCCL_SOCKET_H_

#include <sys/socket.h>
#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <netinet/in.h>
#include <unistd.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <net/if.h>
#include "utils.h"

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SLEEP_INT     1000  // sleep interval in usec
#define RETRY_TIMES   2e4   // retry times before reporting a timeout (20 sec)

/* Common socket address storage structure for IPv4/IPv6 */
union socketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

/* Format a string representation of a (struct sockaddr *) socket address using getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
static inline const char *socketToString(struct sockaddr *saddr, char *buf) {
  if (buf == NULL || saddr == NULL) return NULL;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) { buf[0]='\0'; return buf; }
  char host[NI_MAXHOST], service[NI_MAXSERV];
  (void) getnameinfo(saddr, sizeof(union socketAddress), host, NI_MAXHOST, service, NI_MAXSERV, NI_NUMERICHOST|NI_NUMERICSERV);
  sprintf(buf, "%s<%s>", host, service);
  return buf;
}

static inline short socketToPort(struct sockaddr *saddr) {
  return ntohs(saddr->sa_family == AF_INET ? ((struct sockaddr_in*)saddr)->sin_port : ((struct sockaddr_in6*)saddr)->sin6_port);
}

/* Allow the user to force the IPv4/IPv6 interface selection */
static inline int envSocketFamily(void) {
  int family = -1; // Family selection is not forced, will use first one found
  char* env = getenv("NCCL_SOCKET_FAMILY");
  if (env == NULL)
    return family;

  if (strcmp(env, "AF_INET") == 0)
    family = AF_INET;  // IPv4
  else if (strcmp(env, "AF_INET6") == 0)
    family = AF_INET6; // IPv6
  return family;
}

static int filterInterfaces(const char* prefixList, char* names, union socketAddress *addrs, int sock_family, int maxIfNameSize, int maxIfs) {
  struct netIf userIfs[MAX_IFS];
  int searchNot = prefixList && prefixList[0] == '^';
  if (searchNot) prefixList++;
  int searchExact = prefixList && prefixList[0] == '=';
  if (searchExact) prefixList++;
  int nUserIfs = parseStringList(prefixList, userIfs, MAX_IFS);

  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && found < maxIfs; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;

    TRACE(NCCL_INIT|NCCL_NET,"Found interface %s:%s", interface->ifa_name, socketToString(interface->ifa_addr, line));

    /* Allow the caller to force the socket family type */
    if (sock_family != -1 && family != sock_family)
      continue;

    /* We also need to skip IPv6 loopback interfaces */
    if (family == AF_INET6) {
      struct sockaddr_in6* sa = (struct sockaddr_in6*)(interface->ifa_addr);
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) continue;
    }

    // check against user specified interfaces
    if (!(matchIfList(interface->ifa_name, -1, userIfs, nUserIfs, searchExact) ^ searchNot)) {
      continue;
    }

    // Check that this interface has not already been saved
    // getifaddrs() normal order appears to be; IPv4, IPv6 Global, IPv6 Link
    int duplicate = 0;
    for (int i = 0; i < found; i++) {
      if (strcmp(interface->ifa_name, names+i*maxIfNameSize) == 0) { duplicate = 1; break; }
    }

    if (!duplicate) {
      // Store the interface name
      strncpy(names+found*maxIfNameSize, interface->ifa_name, maxIfNameSize);
      // Store the IP address
      int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
      memcpy(addrs+found, interface->ifa_addr, salen);
      found++;
    }
  }

  freeifaddrs(interfaces);
  return found;
}

static ncclResult_t GetSocketAddrFromString(union socketAddress* ua, const char* ip_port_pair)  __attribute__((unused));
static ncclResult_t GetSocketAddrFromString(union socketAddress* ua, const char* ip_port_pair) {
  if (!(ip_port_pair && strlen(ip_port_pair) > 1)) {
    WARN("Net : string is null");
    return ncclInvalidArgument;
  }

  int ipv6 = ip_port_pair[0] == '[';
  /* Construct the sockaddress structure */
  if (!ipv6) {
    struct netIf ni;
    // parse <ip_or_hostname>:<port> string, expect one pair
    if (parseStringList(ip_port_pair, &ni, 1) != 1) {
      WARN("Net : No valid <IPv4_or_hostname>:<port> pair found");
      return ncclInvalidArgument;
    }

    struct addrinfo hints, *p;
    int rv;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if ( (rv = getaddrinfo(ni.prefix, NULL, &hints, &p)) != 0) {
      WARN("Net : error encountered when getting address info : %s", gai_strerror(rv));
      return ncclInvalidArgument;
    }

    // use the first
    if (p->ai_family == AF_INET) {
      struct sockaddr_in* sin = &ua->sin;
      memcpy(sin, p->ai_addr, sizeof(struct sockaddr_in));
      sin->sin_family = AF_INET;                        // IPv4
      //inet_pton(AF_INET, ni.prefix, &(sin.sin_addr));  // IP address
      sin->sin_port = htons(ni.port);                   // port
    } else if (p->ai_family == AF_INET6) {
      struct sockaddr_in6* sin6 = &ua->sin6;
      memcpy(sin6, p->ai_addr, sizeof(struct sockaddr_in6));
      sin6->sin6_family = AF_INET6;                     // IPv6
      sin6->sin6_port = htons(ni.port);                 // port
      sin6->sin6_flowinfo = 0;                          // needed by IPv6, but possibly obsolete
      sin6->sin6_scope_id = 0;                          // should be global scope, set to 0
    } else {
      WARN("Net : unsupported IP family");
      return ncclInvalidArgument;
    }

    freeaddrinfo(p); // all done with this structure

  } else {
    int i, j = -1, len = strlen(ip_port_pair);
    for (i = 1; i < len; i++) {
      if (ip_port_pair[i] == '%') j = i;
      if (ip_port_pair[i] == ']') break;
    }
    if (i == len) {
      WARN("Net : No valid [IPv6]:port pair found");
      return ncclInvalidArgument;
    }
    int global_scope = (j == -1);     // If no % found, global scope; otherwise, link scope

    char ip_str[NI_MAXHOST], port_str[NI_MAXSERV], if_name[IFNAMSIZ];
    memset(ip_str, '\0', sizeof(ip_str));
    memset(port_str, '\0', sizeof(port_str));
    memset(if_name, '\0', sizeof(if_name));
    strncpy(ip_str, ip_port_pair+1, global_scope ? i-1 : j-1);
    strncpy(port_str, ip_port_pair+i+2, len-i-1);
    int port = atoi(port_str);
    if (!global_scope) strncpy(if_name, ip_port_pair+j+1, i-j-1); // If not global scope, we need the intf name

    struct sockaddr_in6* sin6 = &ua->sin6;
    sin6->sin6_family = AF_INET6;                       // IPv6
    inet_pton(AF_INET6, ip_str, &(sin6->sin6_addr));    // IP address
    sin6->sin6_port = htons(port);                      // port
    sin6->sin6_flowinfo = 0;                            // needed by IPv6, but possibly obsolete
    sin6->sin6_scope_id = global_scope ? 0 : if_nametoindex(if_name);  // 0 if global scope; intf index if link scope
  }
  return ncclSuccess;
}

static int findInterfaces(char* ifNames, union socketAddress *ifAddrs, int ifNameMaxSize, int maxIfs) {
  int nIfs = 0;
  // Allow user to force the INET socket family selection
  int sock_family = envSocketFamily();
  // User specified interface
  char* env = getenv("NCCL_SOCKET_IFNAME");
  if (env && strlen(env) > 1) {
    // Specified by user : find or fail
    nIfs = filterInterfaces(env, ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
  } else {
    // Try to automatically pick the right one
    // Start with IB
    nIfs = filterInterfaces("ib", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
    // Then look for anything else (but not docker or lo)
    if (nIfs == 0) nIfs = filterInterfaces("^docker,lo", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
    // Finally look for docker, then lo.
    if (nIfs == 0) nIfs = filterInterfaces("docker", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
    if (nIfs == 0) nIfs = filterInterfaces("lo", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
  }
  return nIfs;
}

static ncclResult_t createListenSocket(int *fd, union socketAddress *localAddr) {
  /* IPv4/IPv6 support */
  int family = localAddr->sa.sa_family;
  int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);

  /* Create socket and bind it to a port */
  int sockfd = socket(family, SOCK_STREAM, 0);
  if (sockfd == -1) {
    WARN("Net : Socket creation failed : %s", strerror(errno));
    return ncclSystemError;
  }

  if (socketToPort(&localAddr->sa)) {
    // Port is forced by env. Make sure we get the port.
    int opt = 1;
    SYSCHECK(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)), "setsockopt");
#ifdef SO_REUSEPORT /* no SO_REUSEPORT old linux versions */
    SYSCHECK(setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt)), "setsockopt");
#endif
  }

  // localAddr port should be 0 (Any port)
  SYSCHECK(bind(sockfd, &localAddr->sa, salen), "bind");

  /* Get the assigned Port */
  socklen_t size = salen;
  SYSCHECK(getsockname(sockfd, &localAddr->sa, &size), "getsockname");

#ifdef ENABLE_TRACE
  char line[1024];
  TRACE(NCCL_INIT|NCCL_NET,"Listening on socket %s", socketToString(&localAddr->sa, line));
#endif

  /* Put the socket in listen mode
   * NB: The backlog will be silently truncated to the value in /proc/sys/net/core/somaxconn
   */
  SYSCHECK(listen(sockfd, 16384), "listen");
  *fd = sockfd;
  return ncclSuccess;
}

static ncclResult_t connectAddress(int* fd, union socketAddress* remoteAddr) {
  /* IPv4/IPv6 support */
  int family = remoteAddr->sa.sa_family;
  int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);

  /* Connect to a hostname / port */
  *fd = socket(family, SOCK_STREAM, 0);
  if (*fd == -1) {
    WARN("Net : Socket creation failed : %s", strerror(errno));
    return ncclSystemError;
  }

  const int one = 1;
  SYSCHECK(setsockopt(*fd, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(int)), "setsockopt");

  /*  const int bufsize = 128*1024;
    SYSCHECK(setsockopt(*fd, SOL_SOCKET, SO_SNDBUF, (char*)&bufsize, sizeof(int)), "setsockopt");
    SYSCHECK(setsockopt(*fd, SOL_SOCKET, SO_RCVBUF, (char*)&bufsize, sizeof(int)), "setsockopt");*/

  char line[1024];
#ifdef ENABLE_TRACE
  TRACE(NCCL_INIT|NCCL_NET,"Connecting to socket %s", socketToString(&remoteAddr->sa, line));
#endif

  int ret;
  int retries = 0;
retry:
  SYSCHECKSYNC(connect(*fd, &remoteAddr->sa, salen), "connect", ret);
  if (ret == 0) return ncclSuccess;
  if (errno == ECONNREFUSED && ++retries < RETRY_TIMES) {
    INFO(NCCL_ALL,"Call to connect returned %s, retrying", strerror(errno)); \
    usleep(SLEEP_INT);
    goto retry;
  }
  WARN("Connect to %s failed : %s", socketToString(&remoteAddr->sa, line), strerror(errno));
  return ncclSystemError;
}

#define NCCL_SOCKET_SEND 0
#define NCCL_SOCKET_RECV 1
static ncclResult_t socketProgress(int op, int fd, void* ptr, int size, int* offset) {
  int bytes = 0;
  char* data = (char*)ptr;
  do {
    if (op == NCCL_SOCKET_RECV) bytes = recv(fd, data+(*offset), size-(*offset), MSG_DONTWAIT);
    if (op == NCCL_SOCKET_SEND) bytes = send(fd, data+(*offset), size-(*offset), MSG_DONTWAIT);
    if (op == NCCL_SOCKET_RECV && bytes == 0) {
      WARN("Net : Connection closed by remote peer");
      return ncclSystemError;
    }
    if (bytes == -1) {
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        WARN("Call to recv failed : %s", strerror(errno));
        return ncclSystemError;
      } else {
        bytes = 0;
      }
    }
    (*offset) += bytes;
  } while (bytes > 0 && (*offset) < size);
  return ncclSuccess;
}

static ncclResult_t socketWait(int op, int fd, void* ptr, int size, int* offset) {
  while (*offset < size)
    NCCLCHECK(socketProgress(op, fd, ptr, size, offset));
  return ncclSuccess;
}

static ncclResult_t socketSend(int fd, void* ptr, int size) {
  int offset = 0;
  NCCLCHECK(socketWait(NCCL_SOCKET_SEND, fd, ptr, size, &offset));
  return ncclSuccess;
}

static ncclResult_t socketReceive(int fd, void* ptr, int size) {
  int offset = 0;
  NCCLCHECK(socketWait(NCCL_SOCKET_RECV, fd, ptr, size, &offset));
  return ncclSuccess;
}

#endif
