#!/bin/bash -leE

SCRIPT_DIR="$(
    cd "$(dirname "$0")"
    pwd -P
)"
cd "${SCRIPT_DIR}"
# shellcheck source=settings.sh
. "${SCRIPT_DIR}/settings.sh"

cd "${WORKSPACE}"

if ! "${WORKSPACE}/autogen.sh"; then
    echo "ERROR: ${WORKSPACE}/autogen.sh failed"
    echo "FAIL"
    exit 1
fi

if ! "${WORKSPACE}/configure" \
    --prefix="${NCCL_RDMA_SHARP_PLUGINS_DIR}" \
    --with-cuda="${CUDA_HOME}" \
    --with-sharp="${HPCX_SHARP_DIR}"; then
    echo "ERROR: ${WORKSPACE}/configure failed"
    echo "FAIL"
    exit 1
fi

if ! make -j install; then
    echo "ERROR: 'make -j install' failed"
    echo "FAIL"
    exit 1
fi

if [ "$DEBUG" = "true" ]; then
    echo "INFO: ${NCCL_RDMA_SHARP_PLUGINS_DIR}:"
    # For debug purposes
    find "${NCCL_RDMA_SHARP_PLUGINS_DIR}" -type f
fi

echo "PASS"
