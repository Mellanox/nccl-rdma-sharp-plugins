#!/bin/bash -l

set -o pipefail

if [ -n "$DEBUG" ]
then
    set -x
fi

# TODO debug
exit 777

HOSTNAME=`hostname -s`
echo "DEBUG: HOSTNAME = $HOSTNAME"

module load dev/cuda10.0
module load hpcx-gcc

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
echo "DEBUG: SCRIPT_DIR = ${SCRIPT_DIR}"

echo "DEBUG: WORKSPACE = ${WORKSPACE}"

if [ -z "${WORKSPACE}" ]
then
    echo "WARNING: WORKSPACE is empty"
    WORKSPACE=`cd ${SCRIPT_DIR}/../; pwd -P`
    echo "DEBUG: WORKSPACE = ${WORKSPACE}"
fi

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

if [ -z "${ENABLE_PACKAGING}" ]
then
    ENABLE_PACKAGING=0
fi

# Clean
cd ${NCCL_SRC_DIR}
make clean

# Build NCCL
make -j src.build CUDA_HOME=${CUDA_HOME}
if [ $? -ne 0 ]
then
    echo "ERROR: 'make src.build' failed"
    echo "FAIL"
    exit 1
fi

if [ "${ENABLE_PACKAGING}" -eq 1 ]
then
    # Packaging
    #make pkg.debian.build
    #if [ $? -ne 0 ]
    #then
    #   echo "ERROR: 'make pkg.debian.build' failed"
    #    exit 1
    #fi

    #make pkg.redhat.build
    #if [ $? -ne 0 ]
    #then
    #   echo "ERROR: 'make pkg.redhat.build' failed"
    #    exit 1
    #fi

    make pkg.txz.build
    if [ $? -ne 0 ]
    then
        echo "ERROR: 'make pkg.txz.build' failed"
        echo "FAIL"
        exit 1
    fi
fi

echo "PASS"
