#!/bin/bash -leE

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
. ${SCRIPT_DIR}/settings.sh

cd ${WORKSPACE}

${WORKSPACE}/autogen.sh
if [ $? -ne 0 ]
then
    echo "ERROR: ${WORKSPACE}/autogen.sh failed"
    echo "FAIL"
    exit 1
fi

${WORKSPACE}/configure \
    --prefix=${WORKSPACE}/_install \
    --with-cuda=${CUDA_HOME} \
    --with-sharp=${HPCX_SHARP_DIR}
if [ $? -ne 0 ]
then
    echo "ERROR: ${WORKSPACE}/configure failed"
    echo "FAIL"
    exit 1
fi

make -j install
if [ $? -ne 0 ]
then
    echo "ERROR: 'make -j install' failed"
    echo "FAIL"
    exit 1
fi

echo "INFO: ${WORKSPACE}/_install:"
find ${WORKSPACE}/_install -type f

echo "PASS"
