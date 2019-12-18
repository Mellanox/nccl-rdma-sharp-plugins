#!/bin/bash -leE

set -o pipefail

echo "INFO: DEBUG = $DEBUG"

if [ "$DEBUG" = "true" ]
then
    set -x
fi

export CUDA_VER="${CUDA_VER:-10.2}"
echo "INFO: CUDA_VER = ${CUDA_VER}"

module load dev/cuda${CUDA_VER}
module load hpcx-gcc
module load ml/ci-tools

# It is needed to disable nccl_rdma_sharp_plugin libs from HPC-X
export LD_LIBRARY_PATH=`echo $LD_LIBRARY_PATH | sed -e 's|nccl_rdma_sharp_plugin|nccl_rdma_sharp_pluginX|g'`

TOP_DIR="$(git rev-parse --show-toplevel)"
echo "INFO: TOP_DIR = ${TOP_DIR}"

echo "INFO: CUDA_HOME = ${CUDA_HOME}"
echo "INFO: HPCX_SHARP_DIR = ${HPCX_SHARP_DIR}"
echo "INFO: HPCX_DIR = ${HPCX_DIR}"
echo "INFO: WORKSPACE = ${WORKSPACE}"

HOSTNAME=`hostname -s`
echo "INFO: HOSTNAME = $HOSTNAME"

WORKSPACE="${WORKSPACE:-${TOP_DIR}}"

CFG_DIR="${TOP_DIR}/jenkins/cfg"
HOSTFILE=${CFG_DIR}/$HOSTNAME/hostfile

if [ ! -f "${HOSTFILE}" ]
then
    echo "ERROR: ${HOSTFILE} doesn't exist or not accessible"
    echo "FAIL"
    exit 1
fi

CI_DIR="${WORKSPACE}/.ci"

if [ ! -d "${HPCX_DIR}" ]
then
    echo "ERROR: ${HPCX_DIR} does not exist or not accessible"
    echo "FAIL"
    exit 1
fi
