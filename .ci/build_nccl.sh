#!/bin/bash -leE

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
echo "DEBUG: SCRIPT_DIR = ${SCRIPT_DIR}"

. ${SCRIPT_DIR}/settings.sh

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
