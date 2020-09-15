/*************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PARAM_H_
#define NCCL_PARAM_H_

#define NCCL_PARAM(name, env, default_value) \
pthread_mutex_t ncclParamMutex##name = PTHREAD_MUTEX_INITIALIZER; \
int64_t ncclParam##name() { \
  NCCL_STATIC_ASSERT(default_value != -1LL, "default value cannot be -1"); \
  static int64_t value = -1LL; \
  pthread_mutex_lock(&ncclParamMutex##name); \
  if (value == -1LL) { \
    value = default_value; \
    char* str = getenv("NCCL_" env); \
    if (str && strlen(str) > 0) { \
      errno = 0; \
      int64_t v = strtoll(str, NULL, 0); \
      if (errno) { \
        INFO(NCCL_ALL,"Invalid value %s for %s, using default %lu.", str, "NCCL_" env, value); \
      } else { \
        value = v; \
        INFO(NCCL_ALL,"%s set by environment to %lu.", "NCCL_" env, value);  \
      } \
    } \
  } \
  pthread_mutex_unlock(&ncclParamMutex##name); \
  return value; \
}

#endif
