#!/bin/bash -l

if [ -z "$DEBUG" ]
then
    set -x
fi

echo "DEBUG: WORKSPACE = $WORKSPACE"

module load dev/cuda10.0
module load hpcx-sharpv2-gcc

echo "DEBUG: CUDA_HOME = ${CUDA_HOME}"
echo "DEBUG: HPCX_SHARP_DIR = ${HPCX_SHARP_DIR}"

cd $WORKSPACE

$WORKSPACE/autogen.sh
if [ $? -ne 0 ]
then
    echo "ERROR: $WORKSPACE/autogen.sh failed"
    exit 1
fi

$WORKSPACE/configure \
    --prefix=$WORKSPACE/_install \
    --with-cuda=${CUDA_HOME} \
    --with-sharp=${HPCX_SHARP_DIR}
if [ $? -ne 0 ]
then
    echo "ERROR: $WORKSPACE/configure failed"
    exit 1
fi

make -j install
if [ $? -ne 0 ]
then
    echo "ERROR: 'make -j install' failed"
    exit 1
fi

echo "DEBUG: $WORKSPACE/_install:"
find $WORKSPACE/_install -type f
