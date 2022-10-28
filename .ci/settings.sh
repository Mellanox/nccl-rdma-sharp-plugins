#!/bin/bash -leE
# PLUGINS
echo "INFO: DEBUG = $DEBUG"
DEBUG=true
if [ "$DEBUG" = "true" ]
then
    set -x
fi

# W/A for SHARP
# CUDA 10.2 is the latest available version we would like to test, CUDA 10.1 is needed for SHARP
# (due to HPC-X is buitl with CUDA 10.1).
# CUDA 10.2 has priority in the env PATH/LD_LIBRARY_PATH.

# TODO remove use HPC-X which is already inside the image

#module load /hpc/local/etc/modulefiles/dev/cuda-latest
HPCX_UBUNTU_INSTALL_DIR=${HPCX_UBUNTU_INSTALL_DIR:-/hpc/noarch/HPCX/unpacked/hpcx-v2.13-gcc-MLNX_OFED_LINUX-5-ubuntu20.04-cuda11-gdrcopy2-nccl2.12-x86_64/}
module load "${HPCX_UBUNTU_INSTALL_DIR}"/modulefiles/hpcx-ompi
# . "${HPCX_UBUNTU_INSTALL_DIR}/hpcx-init.sh"
# hpcx_load

# It is needed to disable nccl_rdma_sharp_plugin libs from HPC-X
LD_LIBRARY_PATH="${LD_LIBRARY_PATH//nccl_rdma_sharp_plugin/nccl_rdma_sharp_pluginX}"
export LD_LIBRARY_PATH
CUDA_HOME=/usr/local/cuda

export NCCL_RDMA_SHARP_PLUGINS_DIR="${NCCL_RDMA_SHARP_PLUGINS_DIR:-${WORKSPACE}/_install}"
echo "INFO: NCCL_RDMA_SHARP_PLUGINS_DIR = ${NCCL_RDMA_SHARP_PLUGINS_DIR}"

TOP_DIR="$(git rev-parse --show-toplevel)"
echo "INFO: TOP_DIR = ${TOP_DIR}"

echo "INFO: CUDA_VER = ${CUDA_VER}"
echo "INFO: CUDA_HOME = ${CUDA_HOME}"
echo "INFO: HPCX_SHARP_DIR = ${HPCX_SHARP_DIR}"
echo "INFO: HPCX_DIR = ${HPCX_DIR}"
echo "INFO: WORKSPACE = ${WORKSPACE}"

HOSTNAME=$(hostname -s)
echo "INFO: HOSTNAME = $HOSTNAME"

WORKSPACE="${WORKSPACE:-${TOP_DIR}}"
CFG_DIR="${WORKSPACE}/.ci/cfg"
HOSTFILE=${CFG_DIR}/$HOSTNAME/hostfile

if [ ! -f "${HOSTFILE}" ]
then
    echo "ERROR: ${HOSTFILE} doesn't exist or not accessible"
    echo "FAIL"
    exit 1
fi

if [ ! -d "${HPCX_DIR}" ]
then
    echo "ERROR: ${HPCX_DIR} does not exist or not accessible"
    echo "FAIL"
    exit 1
fi
