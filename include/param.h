/*************************************************************************
 * Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PARAM_H_
#define NCCL_PARAM_H_

#include <stdint.h>

const char* userHomeDir();
void setEnvFile(const char* fileName);
void initEnv();
const char *ncclGetEnv(const char *name);

void ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache);

#define NCCL_PARAM(name, env, deftVal) \
  int64_t ncclParam##name() { \
    NCCL_STATIC_ASSERT(deftVal != INT64_MIN, "default value cannot be the uninitialized value."); \
    static int64_t cache = INT64_MIN; \
    if (__builtin_expect(__atomic_load_n(&cache, __ATOMIC_RELAXED) == INT64_MIN, false)) { \
      ncclLoadParam("NCCL_" env, deftVal, INT64_MIN, &cache); \
  } \
    return cache; \
  }

#endif
