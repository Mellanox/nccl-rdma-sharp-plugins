#!/bin/bash -l

if [ -z "$DEBUG" ]
then
    set -x
fi

echo "DEBUG: {SRC_ROOT} = ${SRC_ROOT}"

module load hpcx-gcc

echo "DEBUG: CUDA_HOME = ${CUDA_HOME}"
echo "DEBUG: HPCX_SHARP_DIR = ${HPCX_SHARP_DIR}"

cd ${SRC_ROOT}

${SRC_ROOT}/autogen.sh
if [ $? -ne 0 ]
then
    echo "ERROR: ${SRC_ROOT}/autogen.sh failed"
    echo "FAIL"
    exit 1
fi

${SRC_ROOT}/configure \
    --prefix=${SRC_ROOT}/_install \
    --with-cuda=${CUDA_HOME} \
    --with-sharp=${HPCX_SHARP_DIR}
if [ $? -ne 0 ]
then
    echo "ERROR: ${SRC_ROOT}/configure failed"
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

echo "DEBUG: ${SRC_ROOT}/_install:"
find ${SRC_ROOT}/_install -type f

echo "PASS"
