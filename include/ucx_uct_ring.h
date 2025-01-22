/*************************************************************************
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_UCX_UCT_RING_H_
#define NCCL_UCX_UCT_RING_H_

#include "nccl.h"
#include <assert.h>

#define NCCL_UCT_RING_SIZE (1 << 7)
#define NCCL_UCT_RING_MASK (NCCL_UCT_RING_SIZE - 1)

typedef struct nccl_uct_ring {
  unsigned first;
  unsigned last;
  unsigned size;
  unsigned entry_size;
  int      tag[NCCL_UCT_RING_SIZE];
  void     *entry;
} nccl_uct_ring_t;

static inline ncclResult_t nccl_uct_ring_init(nccl_uct_ring_t *ring,
                                              unsigned entry_size) {
  int i;

  ring->first      = 0;
  ring->last       = 0;
  ring->entry_size = entry_size;
  ring->entry      = malloc(entry_size * NCCL_UCT_RING_SIZE);
  if (ring->entry == NULL) {
    free(ring->entry);
    return ncclSystemError;
  }

  for (i = 0; i < NCCL_UCT_RING_SIZE; i++) {
    ring->tag[i] = INT_MAX;
  }
  return ncclSuccess;
}

static inline void nccl_uct_ring_deinit(nccl_uct_ring_t *ring) {
  free(ring->entry);
}

static inline void *nccl_uct_ring_get_entry(nccl_uct_ring_t *ring, unsigned i) {
  return (uint8_t*)ring->entry + (ring->entry_size * (i & NCCL_UCT_RING_MASK));
}

static inline void nccl_uct_ring_append(nccl_uct_ring_t *ring, int tag,
                                        void *data, size_t len) {
  int j = ring->last & NCCL_UCT_RING_MASK;

  ring->last++;

  assert((ring->last & NCCL_UCT_RING_MASK) !=
         (ring->first & NCCL_UCT_RING_MASK));
  assert(ring->tag[j] == INT_MAX);
  assert(len == ring->entry_size);

  ring->tag[j] = tag;
  memcpy(nccl_uct_ring_get_entry(ring, j), data, len);
}

static inline int nccl_uct_ring_is_empty(const nccl_uct_ring_t *ring) {
  return ring->first == ring->last;
}

static inline void nccl_uct_ring_consume(nccl_uct_ring_t *ring, unsigned i) {
  unsigned j = i & NCCL_UCT_RING_MASK;

  assert(ring->tag[j] != INT_MAX);
  ring->tag[j] = INT_MAX;

  /* Cleanup upon tag hit */
  if (i == ring->first) {
    for (; i != ring->last; i++) {
      j = i & NCCL_UCT_RING_MASK;
      if (ring->tag[j] != INT_MAX) {
        break;
      }
      ring->first = i + 1;
    }
  }
}

static inline unsigned nccl_uct_ring_find(nccl_uct_ring_t *ring, int tag) {
  unsigned i;

  assert(tag != INT_MAX);

  for (i = ring->first; i != ring->last; i++) {
    if (ring->tag[i & NCCL_UCT_RING_MASK] == tag) {
      return i;
    }
  }

  return ring->last;
}

#endif /* NCCL_UCX_UCT_RING_H_ */
