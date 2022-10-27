# nccl-rdma-sharp-plugins Continuous Integration (CI)
## Overview
nccl-rdma-sharp-plugins CI is intended to make sanity checking for every code change. CI is started for each Pull Request (PR) and can be additionally triggered with **bot:mlx:test** (or **bot:mlx:retest**) keyword written in the PR comments. For users in the project WhiteList CI is started automatically, for others - project maintainers should approve CI start with '**ok to test**' keyword reply.<br>
CI status and artefacts (log files) are published within the PR comments.
## Description
CI includes the following steps:
* Build nccl-rdma-sharp-plugins
* Test nccl-rdma-sharp-plugins with [NCCL tests](https://github.com/nvidia/nccl-tests). 
The tests are run with [NVIDIA's NCCL](https://github.com/NVIDIA/nccl) library built within CI from the internal repository.
### Test Environment
CI is run in the Mellanox lab on a 2-node cluster with the following parameters:

Hardware
* IB: 1x ConnectX-6 HCA (connected to Mellanox Quantum™ HDR switch)
* GPU: 1x Nvidia Tesla K40m

Software
* Ubuntu 18.04.4
* Internal stable MLNX_OFED, HPC-X and SHARP versions