# CI tester
# nccl-rdma-sharp-plugins

nccl-rdma-sharp plugin enables RDMA and Switch based collectives(SHARP)
with [NVIDIA's NCCL](https://github.com/NVIDIA/nccl) library

## Overview

## Requirements

* MOFED
* CUDA
* SHARP
* NCCL
* GPUDirectRDMA plugin

## Build Instructions

### build system requirements

* CUDA
* SHARP
* MOFED

Plugin uses GNU autotools for its build system. You can build it as follows:


```
$ ./autogen.sh
$ ./configure
$ make
$ make install
```

The following flags enabled to build with custom dependencies


```
  --with-verbs=PATH       Path to non-standard libibverbs installation
  --with-sharp=PATH       Path to non-standard SHARP installation
  --with-cuda=PATH        Path to non-standard CUDA installation
```


