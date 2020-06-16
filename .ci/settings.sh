#!/bin/bash -leE

echo "INFO: DEBUG = $DEBUG"

if [ "$DEBUG" = "true" ]
then
    set -x
fi

export CUDA_VER="${CUDA_VER:-10.2}"
echo "INFO: CUDA_VER = ${CUDA_VER}"

module load ml/ci-tools

# W/A for SHARP
# CUDA 10.2 is the latest available version we would like to test, CUDA 10.1 is needed for SHARP
# (due to HPC-X is buitl with CUDA 10.1).
# CUDA 10.2 has priority in the env PATH/LD_LIBRARY_PATH.
module load dev/cuda10.1

module load "dev/cuda${CUDA_VER}"

# TODO remove use HPC-X which is already inside the image
#
HPCX_UBUNTU_INSTALL_DIR=${HPCX_UBUNTU_INSTALL_DIR:-/.autodirect/mtrswgwork/artemry/ci_tools_do_not_remove/hpcx-v2.7.pre-gcc-MLNX_OFED_LINUX-5.0-1.0.0.0-ubuntu18.04-x86_64/}
# shellcheck source=/.autodirect/mtrswgwork/artemry/ci_tools_do_not_remove/hpcx-v2.6.0-gcc-MLNX_OFED_LINUX-5.0-1.0.0.0-ubuntu18.04-x86_64/hpcx-init.sh
. "${HPCX_UBUNTU_INSTALL_DIR}/hpcx-init.sh"
hpcx_load

# It is needed to disable nccl_rdma_sharp_plugin libs from HPC-X
LD_LIBRARY_PATH="${LD_LIBRARY_PATH//nccl_rdma_sharp_plugin/nccl_rdma_sharp_pluginX}"
export LD_LIBRARY_PATH

TOP_DIR="$(git rev-parse --show-toplevel)"
echo "INFO: TOP_DIR = ${TOP_DIR}"

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
