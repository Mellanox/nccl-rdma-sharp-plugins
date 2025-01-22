/*************************************************************************
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2016-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SOCKET_H_
#define NCCL_SOCKET_H_

#include "nccl.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <fcntl.h>
#include <poll.h>
#include "stdbool.h"
#include "utils.h"

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SOCKET_NAME_MAXLEN (NI_MAXHOST+NI_MAXSERV)
#define NCCL_SOCKET_MAGIC 0x564ab9f2fc4b9d6cULL

/* Common socket address storage structure for IPv4/IPv6 */
union ncclSocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

enum ncclSocketState {
  ncclSocketStateNone = 0,
  ncclSocketStateInitialized = 1,
  ncclSocketStateAccepting = 2,
  ncclSocketStateAccepted = 3,
  ncclSocketStateConnecting = 4,
  ncclSocketStateConnectPolling = 5,
  ncclSocketStateConnected = 6,
  ncclSocketStateReady = 7,
  ncclSocketStateTerminating = 8,
  ncclSocketStateClosed = 9,
  ncclSocketStateError = 10,
  ncclSocketStateNum = 11

};

enum ncclSocketType {
  ncclSocketTypeUnknown = 0,
  ncclSocketTypeBootstrap = 1,
  ncclSocketTypeProxy = 2,
  ncclSocketTypeNetIb = 4,
  ncclSocketTypeRasNetwork = 5
};

struct ncclSocket {
  int fd;
  int acceptFd;
  int errorRetries;
  union ncclSocketAddress addr;
  volatile uint32_t* abortFlag;
  int asyncFlag;
  enum ncclSocketState state;
  int salen;
  uint64_t magic;
  enum ncclSocketType type;
  int customRetry;
  int finalizeCounter; // Used to keep track of initial handshake for async sockets.
  char finalizeBuffer[sizeof(uint64_t)]; // Used to keep track of initial handshake for async sockets.
};

const char *ncclSocketToString(const union ncclSocketAddress *addr, char *buf, const int numericHostForm);
ncclResult_t ncclSocketGetAddrFromString(union ncclSocketAddress* ua, const char* ip_port_pair);
int ncclFindInterfaceMatchSubnet(char* ifNames, union ncclSocketAddress* localAddrs, union ncclSocketAddress* remoteAddr, int ifNameMaxSize, int maxIfs);
int ncclFindInterfaces(char* ifNames, union ncclSocketAddress *ifAddrs, int ifNameMaxSize, int maxIfs);

// Initialize a socket
ncclResult_t ncclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic, enum ncclSocketType type, volatile uint32_t* abortFlag, int asyncFlag, int customRetry);
// Create a listening socket. sock->addr can be pre-filled with IP & port info. sock->fd is set after a successful call
ncclResult_t ncclSocketListen(struct ncclSocket* sock);
ncclResult_t ncclSocketGetAddr(struct ncclSocket* sock, union ncclSocketAddress* addr);
// Connect to sock->addr. sock->fd is set after a successful call.
ncclResult_t ncclSocketConnect(struct ncclSocket* sock);
// Return socket connection state.
ncclResult_t ncclSocketReady(struct ncclSocket* sock, int *running);
// Accept an incoming connection from listenSock->fd and keep the file descriptor in sock->fd, with the remote side IP/port in sock->addr.
ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket* ulistenSock);
ncclResult_t ncclSocketGetFd(struct ncclSocket* sock, int* fd);
ncclResult_t ncclSocketSetFd(int fd, struct ncclSocket* sock);

#define NCCL_SOCKET_SEND 0
#define NCCL_SOCKET_RECV 1

ncclResult_t ncclSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int* closed);
ncclResult_t ncclSocketWait(int op, struct ncclSocket* sock, void* ptr, int size, int* offset);
ncclResult_t ncclSocketSend(struct ncclSocket* sock, void* ptr, int size);
ncclResult_t ncclSocketRecv(struct ncclSocket* sock, void* ptr, int size);
ncclResult_t ncclSocketTryRecv(struct ncclSocket* sock, void* ptr, int size, int* closed, bool blocking);
ncclResult_t ncclSocketClose(struct ncclSocket* sock);
#endif
