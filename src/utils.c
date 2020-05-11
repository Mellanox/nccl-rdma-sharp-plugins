/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "utils.h"
#include "core.h"
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>
#include <fcntl.h>

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked DONTFORK
// and if they are shared, that could cause a crash in a child process
ncclResult_t ncclIbMalloc(void** ptr, size_t size) {
  size_t page_size = sysconf(_SC_PAGESIZE);
  void* p;
  int size_aligned = ROUNDUP(size, page_size);
  int ret = posix_memalign(&p, page_size, size_aligned);
  if (ret != 0) return ncclSystemError;
  memset(p, 0, size);
  *ptr = p;
  return ncclSuccess;
}

int parseStringList(const char* string, struct netIf* ifList, int maxList) {
  if (!string) return 0;

  const char* ptr = string;

  int ifNum = 0;
  int ifC = 0;
  char c;
  do {
    c = *ptr;
    if (c == ':') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = atoi(ptr+1);
        ifNum++; ifC = 0;
      }
      while (c != ',' && c != '\0') c = *(++ptr);
    } else if (c == ',' || c == '\0') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = -1;
        ifNum++; ifC = 0;
      }
    } else {
      ifList[ifNum].prefix[ifC] = c;
      ifC++;
    }
    ptr++;
  } while (ifNum < maxList && c);
  return ifNum;
}

static int matchIf(const char* string, const char* ref, int matchExact) {
  // Make sure to include '\0' in the exact case
  int matchLen = matchExact ? strlen(string) + 1 : strlen(ref);
  return strncmp(string, ref, matchLen) == 0;
}

static int matchPort(const int port1, const int port2) {
  if (port1 == -1) return 1;
  if (port2 == -1) return 1;
  if (port1 == port2) return 1;
  return 0;
}


int matchIfList(const char* string, int port, struct netIf* ifList, int listSize, int matchExact) {
  // Make an exception for the case where no user list is defined
  if (listSize == 0) return 1;

  for (int i=0; i<listSize; i++) {
    if (matchIf(string, ifList[i].prefix, matchExact)
        && matchPort(port, ifList[i].port)) {
      return 1;
    }
  }
  return 0;
}

static size_t readFileVarArg(char *buffer, size_t max,
    const char *filename_fmt, va_list ap)
{
  char filename[PATH_MAX];
  ssize_t read_bytes;
  int fd;

  vsnprintf(filename, PATH_MAX, filename_fmt, ap);

  fd = open(filename, O_RDONLY);
  if (fd < 0) {
    return -1;
  }

  read_bytes = read(fd, buffer, max - 1);
  if (read_bytes < 0) {
    return -1;
  }

  if (read_bytes < max) {
    buffer[read_bytes] = '\0';
  }

out_close:
  close(fd);
}

int readFileNumber(long *value, const char *filename_fmt, ...)
{
  char buffer[64], *tail;
  ssize_t read_bytes;
  va_list ap;
  long n;

  va_start(ap, filename_fmt);
  read_bytes = readFileVarArg(buffer, sizeof(buffer) - 1,
      filename_fmt, ap);
  va_end(ap);

  if (read_bytes < 0) {
    /* read error */
    return -1;
  }

  n = strtol(buffer, &tail, 0);
  if ((*tail != '\0') && !isspace(*tail)) {
    /* parse error */
    return -1;
  }

  *value = n;
  return 0;
}
