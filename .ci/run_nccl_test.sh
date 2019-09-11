#!/bin/bash -leE

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
. ${SCRIPT_DIR}/settings.sh

GLOBAL_TEST_STATUS=0

if [ -z "${NCCL_DIR}" ]
then
    echo "ERROR: NCCL_DIR is not defined"
    echo "FAIL"
    exit 1
fi

if [ -z "${NCCL_RDMA_SHARP_PLUGINS_DIR}" ]
then
    echo "ERROR: NCCL_RDMA_SHARP_PLUGINS_DIR is not defined"
    echo "FAIL"
    exit 1
fi

if [ -z "${NCCL_TESTS_DIR}" ]
then
    echo "ERROR: NCCL_TESTS_DIR is not defined"
    echo "FAIL"
    exit 1
fi

HOSTFILE=${CI_DIR}/cfg/$HOSTNAME/hostfile
NP=2
IB_DEV="mlx5_0:1"
MPIRUN_OPTIONS_COMMON="\
-x LD_LIBRARY_PATH \
-x NCCL_DEBUG=INFO \
-x HCOLL_MAIN_IB=${IB_DEV} \
-x NCCL_DEBUG_SUBSYS=INIT \
-x NCCL_IB_HCA=${IB_DEV} \
-x NCCL_NET_GDR_LEVEL=5 \
-x NCCL_SOCKET_IFNAME=eno1 \
-x UCX_NET_DEVICES=${IB_DEV} \
-x HCOLL_ENABLE_SHARP=0 \
-x HCOLL_ENABLE_MCAST_ALL=0 \
-mca pml ucx \
-mca coll_hcoll_enable 1 \
--map-by node \
--bind-to none \
--hostfile ${HOSTFILE} \
-np $NP \
--report-bindings \
--allow-run-as-root \
"

# Application options
ITER=100
WARMUP_ITER=100
MSG_SIZE_MIN="8"
MSG_SIZE_MAX="4M"
# MPI_APP="hostname"
MPI_APP="\
${AFFINITY} \
${NCCL_TESTS_DIR}/build/all_reduce_perf \
    -b ${MSG_SIZE_MIN} \
    -e ${MSG_SIZE_MAX} \
    -f 2 \
    -g 1 \
    -c 1 \
    -z 1 \
    -n $ITER \
    -w $WARMUP_ITER \
    -p 0 \
"
ENABLE_SAT=${ENABLE_SAT:-1}
echo "INFO: ENABLE_SAT = ${ENABLE_SAT}"

echo_hash_line() {
    echo "###############################################################################"
}

echo "CUDA_HOME: ${CUDA_HOME}"
echo "NCCL_DIR: ${NCCL_DIR}"
echo "NCCL_RDMA_SHARP_PLUGINS_DIR: ${NCCL_RDMA_SHARP_PLUGINS_DIR}"
echo "MPI_HOME: ${MPI_HOME}"

if [ ! -f "${HOSTFILE}" ]
then
    echo "ERROR: ${HOSTFILE} doesn't exist or not accessible"
    echo "FAIL"
    exit 1
fi

# Build NCCL-TESTS
cd ${NCCL_TESTS_DIR}
make -j clean

make -j CUDA_HOME="${CUDA_HOME}" NCCL_HOME="${NCCL_DIR}" MPI=1 MPI_HOME="${MPI_HOME}"

export LD_LIBRARY_PATH="${NCCL_DIR}/lib:${NCCL_RDMA_SHARP_PLUGINS_DIR}/lib:${LD_LIBRARY_PATH}"

# USAGE: all_reduce_perf
        # [-t,--nthreads <num threads>]
        # [-g,--ngpus <gpus per thread>]
        # [-b,--minbytes <min size in bytes>]
        # [-e,--maxbytes <max size in bytes>]
        # [-i,--stepbytes <increment size>]
        # [-f,--stepfactor <increment factor>]
        # [-n,--iters <iteration count>]
        # [-m,--agg_iters <aggregated iteration count>]
        # [-w,--warmup_iters <warmup iteration count>]
        # [-p,--parallel_init <0/1>]
        # [-c,--check <0/1>]
        # [-o,--op <sum/prod/min/max/all>]
        # [-d,--datatype <nccltype/all>]
        # [-r,--root <root>]
        # [-z,--blocking <0/1>]
        # [-h,--help]

###############################################################################
# Run NCCL-TESTS (MPI)
###############################################################################

###############################################################################
# Test 1
###############################################################################
(
    echo_hash_line
    echo "# Test 1..."
    echo_hash_line

    MPIRUN_OPTIONS_SPECIFIC="\
    -x NCCL_LL_THRESHOLD=0 \
    -x NCCL_TREE_THRESHOLD=1000000000 \
    -x SHARP_COLL_LOG_LEVEL=3 \
    -x ENABLE_SHARP_COLL=1 \
    -x SHARP_COLL_OSTS_PER_GROUP=64 \
    -x SHARP_COLL_ENABLE_MCAST_TARGET=0 \
    -x SHARP_COLL_JOB_QUOTA_PAYLOAD_PER_OST=1024 \
    -x SHARP_COLL_JOB_QUOTA_OSTS=256 \
    -x SHARP_COLL_ENABLE_SAT=${ENABLE_SAT} \
    -x SHARP_COLL_SAT_THRESHOLD=1 \
    "
    mpirun \
        ${MPIRUN_OPTIONS_COMMON} \
        ${MPIRUN_OPTIONS_SPECIFIC} \
        ${MPI_APP}
)
if [ $? -ne 0 ]
then
    echo "# Test 1... failed"
    GLOBAL_TEST_STATUS=1
else
    echo "# Test 1... passed"
fi

###############################################################################
# Test 2
###############################################################################
(
    echo_hash_line
    echo "# Test 2..."
    echo_hash_line

    MPIRUN_OPTIONS_SPECIFIC="\
    -x NCCL_LL_THRESHOLD=1000000000 \
    -x NCCL_TREE_THRESHOLD=1000000000 \
    -x SHARP_COLL_LOG_LEVEL=3 \
    -x ENABLE_SHARP_COLL=1 \
    -x SHARP_COLL_OSTS_PER_GROUP=64 \
    -x SHARP_COLL_ENABLE_MCAST_TARGET=0 \
    -x SHARP_COLL_JOB_QUOTA_PAYLOAD_PER_OST=1024 \
    -x SHARP_COLL_JOB_QUOTA_OSTS=256 \
    -x SHARP_COLL_ENABLE_SAT=$ENABLE_SAT \
    -x SHARP_COLL_SAT_THRESHOLD=1 \
    "

    mpirun \
        ${MPIRUN_OPTIONS_COMMON} \
        ${MPIRUN_OPTIONS_SPECIFIC} \
        ${MPI_APP}
)
if [ $? -ne 0 ]
then
    echo "# Test 2... failed"
    GLOBAL_TEST_STATUS=1
else
    echo "# Test 2... passed"
fi

###############################################################################
# Test 3
###############################################################################
(
    echo_hash_line
    echo "# Test 3..."
    echo_hash_line

    MPIRUN_OPTIONS_SPECIFIC="\
    -x NCCL_LL_THRESHOLD=1000000000 \
    -x NCCL_TREE_THRESHOLD=1000000000 \
    "
    mpirun \
        ${MPIRUN_OPTIONS_COMMON} \
        ${MPIRUN_OPTIONS_SPECIFIC} \
        ${MPI_APP}
)
if [ $? -ne 0 ]
then
    echo "# Test 3... failed"
    GLOBAL_TEST_STATUS=1
else
    echo "# Test 3... passed"
fi

###############################################################################
# Test 4
###############################################################################
(
    echo_hash_line
    echo "# Test 4..."
    echo_hash_line
    
    MPIRUN_OPTIONS_SPECIFIC="\
    -x NCCL_LL_THRESHOLD=0 \
    -x NCCL_TREE_THRESHOLD=1000000000 \
    "
    mpirun \
        ${MPIRUN_OPTIONS_COMMON} \
        ${MPIRUN_OPTIONS_SPECIFIC} \
        ${MPI_APP}
)
if [ $? -ne 0 ]
then
    echo "# Test 4... failed"
    GLOBAL_TEST_STATUS=1
else
    echo "# Test 4... passed"
fi

###############################################################################
# Test 5
###############################################################################
(
    echo_hash_line
    echo "# Test 5..."
    echo_hash_line
    
    MPIRUN_OPTIONS_SPECIFIC="\
    -x NCCL_TREE_THRESHOLD=1000000000 \
    "
    mpirun \
        ${MPIRUN_OPTIONS_COMMON} \
        ${MPIRUN_OPTIONS_SPECIFIC} \
        ${MPI_APP}
)
if [ $? -ne 0 ]
then
    echo "# Test 5... failed"
    GLOBAL_TEST_STATUS=1
else
    echo "# Test 5... passed"
fi

###############################################################################
# Test 6
###############################################################################
(
    echo_hash_line
    echo "# Test 6..."
    echo_hash_line
    
    MPIRUN_OPTIONS_SPECIFIC="\
    -x NCCL_LL_THRESHOLD=1000000000 \
    -x NCCL_TREE_THRESHOLD=0 \
    "
    mpirun \
        ${MPIRUN_OPTIONS_COMMON} \
        ${MPIRUN_OPTIONS_SPECIFIC} \
        ${MPI_APP}
)
if [ $? -ne 0 ]
then
    echo "# Test 6... failed"
    GLOBAL_TEST_STATUS=1
else
    echo "# Test 6... passed"
fi

###############################################################################
# Test 7
###############################################################################
(
    echo_hash_line
    echo "# Test 7..."
    echo_hash_line
    
    MPIRUN_OPTIONS_SPECIFIC="\
    -x NCCL_LL_THRESHOLD=0 \
    -x NCCL_TREE_THRESHOLD=0 \
    "
    mpirun \
        ${MPIRUN_OPTIONS_COMMON} \
        ${MPIRUN_OPTIONS_SPECIFIC} \
        ${MPI_APP}
)
if [ $? -ne 0 ]
then
    echo "# Test 7... failed"
    GLOBAL_TEST_STATUS=1
else
    echo "# Test 7... passed"
fi

###############################################################################
# Test 8
###############################################################################
(
    echo_hash_line
    echo "# Test 8..."
    echo_hash_line
    
    MPIRUN_OPTIONS_SPECIFIC="\
    -x NCCL_TREE_THRESHOLD=0 \
    "
    mpirun \
        ${MPIRUN_OPTIONS_COMMON} \
        ${MPIRUN_OPTIONS_SPECIFIC} \
        ${MPI_APP}
)
if [ $? -ne 0 ]
then
    echo "# Test 8... failed"
    GLOBAL_TEST_STATUS=1
else
    echo "# Test 8... passed"
fi

###############################################################################
if [ ${GLOBAL_TEST_STATUS} -ne 0 ]
then
    echo "ERROR: some tests failed, check the log file"
    echo "FAIL"
    exit 1
else
    echo "All tests PASSED"
fi

echo "PASS"
