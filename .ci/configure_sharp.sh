#!/bin/bash -l

SCRIPT_DIR="$( cd "$(dirname "$0")" || exit 1 ; pwd -P )"
cd "${SCRIPT_DIR}" || exit 1
# shellcheck source=settings.sh
. "${SCRIPT_DIR}/settings.sh"

if [ -z "${NCCL_RDMA_SHARP_PLUGINS_DIR}" ]
then
    echo "ERROR: NCCL_RDMA_SHARP_PLUGINS_DIR is not defined"
    echo "FAIL"
    exit 1
fi

export LD_LIBRARY_PATH="${NCCL_RDMA_SHARP_PLUGINS_DIR}/lib:${LD_LIBRARY_PATH}"

# 1 - run sanity tests, 0 - do not run
VERIFY_SHARP_ENABLE=${VERIFY_SHARP_ENABLE:-1}

if [ -z "${NCCL_DIR}" ]
then
    module load dev/nccl-nightly-stable
else
    export LD_LIBRARY_PATH="${NCCL_DIR}/lib:${LD_LIBRARY_PATH}"
fi

# Available values: start|stop|restart
SHARP_MANAGER_ACTION="${1:-restart}"
echo "INFO: SHARP_MANAGER_ACTION = ${SHARP_MANAGER_ACTION}"

echo "INFO: NFS_WORKSPACE = ${NFS_WORKSPACE}"

if [ -z "${NFS_WORKSPACE}" ]
then
    echo "ERROR: NFS_WORKSPACE is not defined"
    echo "FAIL"
    exit 1
fi

if [ -z "${HPCX_SHARP_DIR}" ]
then
    echo "ERROR: HPCX_SHARP_DIR is not defined"
    echo "FAIL"
    exit 1
fi

CONFIGURE_SHARP_TMP_DIR="${NFS_WORKSPACE}/configure_sharp_$$"
mkdir -p "${CONFIGURE_SHARP_TMP_DIR}"

export SHARP_CONF="${CONFIGURE_SHARP_TMP_DIR}"
export SHARP_INI_FILE="${SHARP_CONF}/sharp_manager.ini"

cp -R "${CFG_DIR}/$HOSTNAME/sharp_conf/"* "${SHARP_CONF}"

if [ -f "${SHARP_CONF}/sharp_am_node.txt" ]
then
    SHARP_AM_NODE=`cat ${SHARP_CONF}/sharp_am_node.txt`
    echo "INFO: SHARP_AM_NODE = ${SHARP_AM_NODE}"
else
    echo "ERROR: ${SHARP_CONF}/sharp_am_node.txt does not exist or not accessible"
    echo "FAIL"
    exit 1
fi

IB_DEV="mlx5_0"
SM_GUID=`sudo sminfo -C ${IB_DEV} -P1 | awk '{print $7}' | cut -d',' -f1`
# SM/AM node
# SM_HOSTNAME=`sudo ibnetdiscover -H -C mlx5_0 -P1  | grep ${SM_GUID} | awk -F'"' '{print $2 }' | awk '{print $1}'`
HOSTS=`cat $HOSTFILE | xargs | tr ' ' ','`

echo "INFO: IB_DEV = ${IB_DEV}"
echo "INFO: SM_GUID = ${SM_GUID}"
# echo "INFO: SM_HOSTNAME = ${SM_HOSTNAME}"
echo "INFO: HOSTS = ${HOSTS}"

rm -f ${SHARP_INI_FILE}

cat > ${SHARP_INI_FILE} <<EOF
sharp_AM_server="${SHARP_AM_NODE}"
sharp_am_log_verbosity="3"
sharp_hostlist="$HOSTS"
sharp_manager_general_conf="${SHARP_CONF}"
sharpd_log_verbosity="3"
EOF

echo "INFO: SHARP_INI_FILE ${SHARP_INI_FILE} BEGIN"
cat ${SHARP_INI_FILE}
echo "INFO: SHARP_INI_FILE ${SHARP_INI_FILE} END"

trim_multiple_spaces() {
    echo "$1" | sed -s "s|\ \ *| |g"
}

check_opensm_status() {
    echo "Checking OpenSM status on ${SHARP_AM_NODE}..."

    ssh "${SHARP_AM_NODE}" "systemctl status opensmd"
    if [ $? -ne 0 ]
    then
        echo "ERROR: opensmd is not run on ${SHARP_AM_NODE}"
        echo "FAIL"
        exit 1
    fi

    echo "Checking OpenSM status on ${SHARP_AM_NODE}... DONE"
}

check_opensm_conf() {
    echo "INFO: check_opensm_conf on ${SHARP_AM_NODE}..."

    OPENSM_CONFIG="/etc/opensm/opensm.conf"
    echo "INFO: opensm config = ${OPENSM_CONFIG}"

    ssh "${SHARP_AM_NODE}" "grep \"routing_engine.*updn\" ${OPENSM_CONFIG} 2>/dev/null"
    if [ $? -ne 0 ]
    then
        echo "ERROR: wrong value of routing_engine parameter in ${OPENSM_CONFIG}"
        echo "Should be (example): routing_engine updn"
        echo "FAIL"
        exit 1
    fi

    ssh "${SHARP_AM_NODE}" "grep \"sharp_enabled.*2\" ${OPENSM_CONFIG} 2>/dev/null"
    if [ $? -ne 0 ]
    then
        echo "ERROR: wrong value of sharp_enabled parameter in ${OPENSM_CONFIG}"
        echo "Should be (example): sharp_enabled 2"
        echo "FAIL"
        exit 1
    fi

    echo "INFO: check_opensm_conf on ${SHARP_AM_NODE}... DONE"
}

verify_sharp() {
    echo "INFO: verify_sharp..."

    cp ${HPCX_SHARP_DIR}/share/sharp/examples/mpi/coll/* ${CONFIGURE_SHARP_TMP_DIR}
    cd ${CONFIGURE_SHARP_TMP_DIR}
    make CUDA=1 CUDA_HOME=${CUDA_HOME} SHARP_HOME="${HPCX_SHARP_DIR}"
    if [ $? -ne 0 ]
    then
        echo "ERROR: verify_sharp make failed"
        echo "FAIL"
        exit 1
    fi

    ITERS=100
    SKIP=20
    NP=$(wc --lines "$HOSTFILE" | awk '{print $1}')

    # -mca coll_hcoll_enable 0 - disable HCOLL
    MPIRUN_COMMON_OPTIONS="\
        -np $NP \
        -H $HOSTS \
        --map-by node \
        -x LD_LIBRARY_PATH \
        --allow-run-as-root \
    "

    # TODO change to SHARP_COLL_SAT_THRESHOLD=1 (32 - W/A for SHARP issue)
    MPIRUN_SHARP_OPTIONS="\
        -x SHARP_COLL_LOG_LEVEL=3 \
        -x ENABLE_SHARP_COLL=1 \
        -x SHARP_COLL_SAT_THRESHOLD=32 \
        -x SHARP_COLL_ENABLE_SAT=1 \
    "

    echo "Environment for the reproducer:"
    echo "export PATH=$PATH"
    echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    echo "export OPAL_PREFIX=${OPAL_PREFIX}"

    # Test 1 (from ${HPCX_SHARP_DIR}/share/sharp/examples/mpi/coll/README):
    # Run allreduce barrier perf test on 2 hosts using port mlx5_0
    echo "Test 1..."
    CMD="mpirun \
            ${MPIRUN_COMMON_OPTIONS} \
            ${MPIRUN_SHARP_OPTIONS} \
            ${CONFIGURE_SHARP_TMP_DIR}/sharp_coll_test \
                -d mlx5_0:1 \
                --iters $ITERS \
                --skip $SKIP \
                --mode perf \
                --collectives allreduce,barrier"
    echo "INFO: Test 1 command line:"
    trim_multiple_spaces "$CMD"
    $CMD
    if [ $? -ne 0 ]
    then
        echo "ERROR: verify_sharp Test 1 failed"
        echo "FAIL"
        exit 1
    fi
    echo "Test 1... DONE"

    # Test 2 (from ${HPCX_SHARP_DIR}/share/sharp/examples/mpi/coll/README):
    # Run allreduce perf test on 2 hosts using port mlx5_0 with CUDA buffers
    echo "Test 2..."
    CMD="mpirun \
            ${MPIRUN_COMMON_OPTIONS} \
            ${MPIRUN_SHARP_OPTIONS} \
            ${CONFIGURE_SHARP_TMP_DIR}/sharp_coll_test \
                -d mlx5_0:1 \
                --iters $ITERS \
                --skip $SKIP \
                --mode perf \
                --collectives allreduce \
                -M cuda"
    echo "INFO: Test 2 command line:"
    trim_multiple_spaces "$CMD"
    $CMD
    if [ $? -ne 0 ]
    then
        echo "ERROR: verify_sharp Test 2 failed"
        echo "FAIL"
        exit 1
    fi
    echo "Test 2... DONE"

    # Test 3 (from ${HPCX_SHARP_DIR}/share/sharp/examples/mpi/coll/README):
    # Run allreduce perf test on 2 hosts using port mlx5_0 with Streaming aggregation from 4B to 512MB
    echo "Test 3..."
    CMD="mpirun \
            ${MPIRUN_COMMON_OPTIONS} \
            ${MPIRUN_SHARP_OPTIONS} \
            ${CONFIGURE_SHARP_TMP_DIR}/sharp_coll_test \
                -d mlx5_0:1 \
                --iters $ITERS \
                --skip $SKIP \
                --mode perf \
                --collectives allreduce \
                -s 4:536870912"
    echo "INFO: Test 3 command line:"
    trim_multiple_spaces "$CMD"
    $CMD
    if [ $? -ne 0 ]
    then
        echo "ERROR: verify_sharp Test 3 failed"
        echo "FAIL"
        exit 1
    fi
    echo "Test 3... DONE"

    # Test 4:
    # Run iallreduce perf test on 2 hosts using port mlx5_0
    echo "Test 4..."
    CMD="mpirun \
            ${MPIRUN_COMMON_OPTIONS} \
            ${MPIRUN_SHARP_OPTIONS} \
            ${CONFIGURE_SHARP_TMP_DIR}/sharp_coll_test \
                -d mlx5_0:1 \
                --iters $ITERS \
                --skip $SKIP \
                --mode perf \
                --collectives iallreduce \
                -N 128"
    echo "INFO: Test 4 command line:"
    trim_multiple_spaces "$CMD"
    $CMD
    if [ $? -ne 0 ]
    then
        echo "ERROR: verify_sharp Test 4 failed"
        echo "FAIL"
        exit 1
    fi
    echo "Test 4... DONE"

    # Test 5:
    # Run iallreduce perf test on 2 hosts using port mlx5_0 with CUDA buffers
    echo "Test 5..."
    CMD="mpirun \
            ${MPIRUN_COMMON_OPTIONS} \
            ${MPIRUN_SHARP_OPTIONS} \
            ${CONFIGURE_SHARP_TMP_DIR}/sharp_coll_test \
                -d mlx5_0:1 \
                --iters $ITERS \
                --skip $SKIP \
                --mode perf \
                --collectives iallreduce \
                -N 128 \
                -M cuda"
    echo "INFO: Test 5 command line:"
    trim_multiple_spaces "$CMD"
    $CMD
    if [ $? -ne 0 ]
    then
        echo "ERROR: verify_sharp Test 5 failed"
        echo "FAIL"
        exit 1
    fi
    echo "Test 5... DONE"

    # Test 6:
    # Run iallreduce perf test on 2 hosts using port mlx5_0 with Streaming aggregation from 4B to 512MB
    echo "Test 6..."
    CMD="mpirun \
            ${MPIRUN_COMMON_OPTIONS} \
            ${MPIRUN_SHARP_OPTIONS} \
            ${CONFIGURE_SHARP_TMP_DIR}/sharp_coll_test \
                -d mlx5_0:1 \
                --iters $ITERS \
                --skip $SKIP \
                --mode perf \
                --collectives iallreduce \
                -N 128 \
                -s 4:131072"
    echo "INFO: Test 6 command line:"
    trim_multiple_spaces "$CMD"
    $CMD
    if [ $? -ne 0 ]
    then
        echo "ERROR: verify_sharp Test 6 failed"
        echo "FAIL"
        exit 1
    fi
    echo "Test 6... DONE"

    # Test 7 (from the SHARP deployment guide): Without SAT
    echo "Test 7..."
    CMD="$OMPI_HOME/bin/mpirun \
            ${MPIRUN_COMMON_OPTIONS} \
            --bind-to core \
            -mca btl_openib_warn_default_gid_prefix 0 \
            -mca rmaps_dist_device mlx5_0:1 \
            -mca rmaps_base_mapping_policy dist:span \
            -x MXM_RDMA_PORTS=mlx5_0:1 \
            -x HCOLL_MAIN_IB=mlx5_0:1 \
            -x MXM_ASYNC_INTERVAL=1800s \
            -x MXM_LOG_LEVEL=ERROR \
            -x HCOLL_ML_DISABLE_REDUCE=1 \
            -x HCOLL_ENABLE_MCAST_ALL=1 \
            -x HCOLL_MCAST_NP=1 \
            -x LD_LIBRARY_PATH \
            -x HCOLL_ENABLE_SHARP=2 \
            -x SHARP_COLL_LOG_LEVEL=3 \
            -x SHARP_COLL_GROUP_RESOURCE_POLICY=1 \
            -x SHARP_COLL_MAX_PAYLOAD_SIZE=256 \
            -x HCOLL_SHARP_UPROGRESS_NUM_POLLS=999 \
            -x HCOLL_BCOL_P2P_ALLREDUCE_SHARP_MAX=4096 \
            -x SHARP_COLL_PIPELINE_DEPTH=32 \
            -x SHARP_COLL_JOB_QUOTA_OSTS=32 \
            -x SHARP_COLL_JOB_QUOTA_MAX_GROUPS=4 \
            -x SHARP_COLL_JOB_QUOTA_PAYLOAD_PER_OST=256 \
            taskset -c 1 \
                numactl --membind=0 \
                    $HPCX_OSU_DIR/osu_allreduce \
                        -i 100 \
                        -x 100 \
                        -f \
                        -m 4096:4096"
    echo "INFO: Test 7 command line:"
    trim_multiple_spaces "$CMD"
    $CMD
    if [ $? -ne 0 ]
    then
        echo "ERROR: Test 7 (without SAT) failed, check the log file"
        echo "FAIL"
        exit 1
    fi
    echo "Test 7... DONE"

    # Test 8 (from the SHARP deployment guide): With SAT
    echo "Test 8..."
    CMD="$OMPI_HOME/bin/mpirun \
            ${MPIRUN_COMMON_OPTIONS} \
            -mca btl_openib_warn_default_gid_prefix 0 \
            -mca rmaps_dist_device mlx5_0:1 \
            -mca rmaps_base_mapping_policy dist:span \
            -x MXM_RDMA_PORTS=mlx5_0:1 \
            -x HCOLL_MAIN_IB=mlx5_0:1 \
            -x MXM_ASYNC_INTERVAL=1800s \
            -x MXM_LOG_LEVEL=ERROR \
            -x HCOLL_ML_DISABLE_REDUCE=1 \
            -x HCOLL_ENABLE_MCAST_ALL=1 \
            -x HCOLL_MCAST_NP=1 \
            -x LD_LIBRARY_PATH \
            -x HCOLL_ENABLE_SHARP=2 \
            -x SHARP_COLL_LOG_LEVEL=3 \
            -x SHARP_COLL_GROUP_RESOURCE_POLICY=1 \
            -x SHARP_COLL_MAX_PAYLOAD_SIZE=256 \
            -x HCOLL_SHARP_UPROGRESS_NUM_POLLS=999 \
            -x HCOLL_BCOL_P2P_ALLREDUCE_SHARP_MAX=4096 \
            -x SHARP_COLL_PIPELINE_DEPTH=32 \
            -x SHARP_COLL_JOB_QUOTA_OSTS=32 \
            -x SHARP_COLL_JOB_QUOTA_MAX_GROUPS=4 \
            -x SHARP_COLL_JOB_QUOTA_PAYLOAD_PER_OST=256 \
            -x SHARP_COLL_ENABLE_SAT=1 \
            taskset -c 1 \
                numactl --membind=0 \
                    $HPCX_OSU_DIR/osu_allreduce \
                        -i 100 \
                        -x 100 \
                        -f \
                    -m 4096:4096"
    echo "INFO: Test 8 command line:"
    trim_multiple_spaces "$CMD"
    $CMD
    if [ $? -ne 0 ]
    then
        echo "ERROR: Test 8 (with SAT) failed, check the log file"
        echo "FAIL"
        exit 1
    fi
    echo "Test 8... DONE"

    echo "INFO: verify_sharp... DONE"
}

if [ "${SHARP_MANAGER_ACTION}" != "stop" ]
then
    check_opensm_status
    check_opensm_conf
fi

sudo PDSH_RCMD_TYPE=ssh SHARP_INI_FILE=${SHARP_INI_FILE} SHARP_CONF=${SHARP_CONF} ${HPCX_SHARP_DIR}/sbin/sharp_manager.sh "${SHARP_MANAGER_ACTION}" -l "$HOSTS" -s "${SHARP_AM_NODE}"
if [ $? -ne 0 ]
then
    echo "ERROR: sharp_manager.sh failed, check the log file"
    echo "FAIL"
    exit 1
fi

if [ "${SHARP_MANAGER_ACTION}" != "stop" ] && [ "${VERIFY_SHARP_ENABLE}" -eq 1 ]
then
    verify_sharp
fi

sudo chmod -R 777 ${CONFIGURE_SHARP_TMP_DIR}
rm -rf ${CONFIGURE_SHARP_TMP_DIR}

echo "PASS"
