#!/bin/bash -leE

set -o pipefail

if [ -n "$DEBUG" ]
then
    set -x
fi

module load dev/cuda10.0
module load hpcx-gcc

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
echo "DEBUG: SCRIPT_DIR = ${SCRIPT_DIR}"

echo "DEBUG: CUDA_HOME = ${CUDA_HOME}"
echo "DEBUG: HPCX_SHARP_DIR = ${HPCX_SHARP_DIR}"
echo "DEBUG: HPCX_DIR = ${HPCX_DIR}"
echo "DEBUG: WORKSPACE = ${WORKSPACE}"

HOSTNAME=`hostname -s`
echo "DEBUG: HOSTNAME = $HOSTNAME"

if [ -z "${WORKSPACE}" ]
then
    echo "WARNING: WORKSPACE is empty"
    WORKSPACE=`cd ${SCRIPT_DIR}/../; pwd -P`
    echo "DEBUG: WORKSPACE = ${WORKSPACE}"
fi

CI_DIR="${WORKSPACE}/.ci"
NCCL_PLUGIN_DIR="${WORKSPACE}/_install"

if [ -z "${SHARP_DIR}" ]
then
    if [ -z "${HPCX_SHARP_DIR}" ]
    then
        echo "ERROR: SHARP_DIR and HPCX_SHARP_DIR not set"
        echo "FAIL"
        exit 1
    else
        SHARP_DIR="${HPCX_SHARP_DIR}"
    fi
fi

echo "DEBUG: SHARP_DIR = ${SHARP_DIR}"

if [ ! -d "${HPCX_DIR}" ]
then
    echo "ERROR: ${HPCX_DIR} does not exist or not accessible"
    echo "FAIL"
    exit 1
fi
