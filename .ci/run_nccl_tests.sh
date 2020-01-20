#!/bin/bash -leE

SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd ${SCRIPT_DIR}
. ${SCRIPT_DIR}/settings.sh

GLOBAL_TEST_STATUS=0

if [ -z "${NCCL_DIR}" ]
then
    module load dev/nccl-nightly-stable
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

NP=2
IB_DEV="mlx5_0:1"

# UCX_MEMTYPE_CACHE=n - to avoid warnings "memtype_cache.c:83   UCX  ERROR failed to insert region 0x1a1e890 [0x7f8d00000000..0x7f8d30000000]: Element already exists"
MPIRUN_OPTIONS_COMMON="\
-x LD_LIBRARY_PATH \
-x NCCL_DEBUG=INFO \
-x HCOLL_MAIN_IB=${IB_DEV} \
-x NCCL_DEBUG_SUBSYS=INIT \
-x NCCL_IB_HCA=${IB_DEV} \
-x NCCL_SOCKET_IFNAME=eno1 \
-x UCX_NET_DEVICES=${IB_DEV} \
-x UCX_MEMTYPE_CACHE=n \
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

# Build NCCL-TESTS
cd ${NCCL_TESTS_DIR}
make -j clean

make -j CUDA_HOME="${CUDA_HOME}" NCCL_HOME="${NCCL_DIR}" MPI=1 MPI_HOME="${MPI_HOME}"

export LD_LIBRARY_PATH="${NCCL_DIR}/lib:${NCCL_RDMA_SHARP_PLUGINS_DIR}/lib:${LD_LIBRARY_PATH}"

trim_multiple_spaces() {
    echo "$1" | sed -s "s|\ \ *| |g"
}

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

i=1
#===================
# P2P plugin
#===================
for PLUGIN_P2P in IB UCX_RNDV
do
    if [ "${PLUGIN_P2P}" = "IB" ]
    then
        MPIRUN_OPTIONS_PLUGIN_P2P="-x NCCL_PLUGIN_P2P=ib"
    elif [ "${PLUGIN_P2P}" = "UCX_HWTM" ]
    then
        MPIRUN_OPTIONS_PLUGIN_P2P="\
            -x NCCL_PLUGIN_P2P=ucx \
            -x UCX_TLS=rc_x,cuda_copy \
            -x UCX_RC_MLX5_TM_SEG_SIZE=2M \
            -x UCX_RC_MLX5_RX_QUEUE_LEN=512 \
            -x UCX_TM_THRESH=1 \
            -x UCX_RNDV_THRESH=inf \
            -x UCX_RNDV_SCHEME=get_zcopy \
            -x UCX_RC_MLX5_TM_ENABLE=y \
            -x UCX_MEMTYPE_CACHE=n \
            "
    else
        MPIRUN_OPTIONS_PLUGIN_P2P="\
            -x NCCL_PLUGIN_P2P=ucx \
            -x UCX_TLS=rc_x,cuda_copy \
            -x UCX_RNDV_THRESH=0 \
            -x UCX_RNDV_SCHEME=get_zcopy \
            -x UCX_MEMTYPE_CACHE=n \
            "
    fi
    #===================
    # NCCL_PROTO
    #===================
    for NCCL_PROTO in Simple LL DEFAULT
    do
        if [ "${NCCL_PROTO}" = "DEFAULT" ]
        then
            MPIRUN_OPTIONS_NCCL_PROTO=""
        else
            MPIRUN_OPTIONS_NCCL_PROTO="-x NCCL_PROTO=${NCCL_PROTO}"
        fi

        #===================
        # NCCL_ALGO
        #===================
        for NCCL_ALGO in CollNet Tree Ring DEFAULT
        do
            if [ "${NCCL_ALGO}" = "DEFAULT" ]
            then
                MPIRUN_OPTIONS_NCCL_ALGO=""
            else
                MPIRUN_OPTIONS_NCCL_ALGO="-x NCCL_ALGO=${NCCL_ALGO}"
            fi

            if [ "${NCCL_ALGO}" = "CollNet" ]
            then
                MPIRUN_OPTIONS_NCCL_ALGO="-x NCCL_COLLNET_ENABLE=1 ${MPIRUN_OPTIONS_NCCL_ALGO}"
            fi

            #===================
            # SHARP_ENABLE
            #===================
            for SHARP_ENABLE in 0 1
            do
                if [ "${SHARP_ENABLE}" = "0" ]
                then
                    MPIRUN_OPTIONS_SHARP=""
                else
                    MPIRUN_OPTIONS_SHARP="\
                        -x SHARP_COLL_LOG_LEVEL=3 \
                        -x ENABLE_SHARP_COLL=1 \
                        -x SHARP_COLL_OSTS_PER_GROUP=64 \
                        -x SHARP_COLL_ENABLE_MCAST_TARGET=0 \
                        -x SHARP_COLL_JOB_QUOTA_PAYLOAD_PER_OST=1024 \
                        -x SHARP_COLL_JOB_QUOTA_OSTS=256 \
                        -x SHARP_COLL_ENABLE_SAT=${ENABLE_SAT} \
                        -x SHARP_COLL_SAT_THRESHOLD=1 \
                        "
                fi

                #===================
                # NCCL_NET_GDR_LEVEL
                #===================
                # for NCCL_NET_GDR_LEVEL in 0 1 2 3 4 5 DEFAULT
                for NCCL_NET_GDR_LEVEL in DEFAULT
                do
                    if [ "${NCCL_NET_GDR_LEVEL}" = "DEFAULT" ]
                    then
                        MPIRUN_OPTIONS_GDR_LEVEL=""
                    else
                        MPIRUN_OPTIONS_GDR_LEVEL="-x NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL}"
                    fi

                    #===================
                    # NCCL_NET_GDR_READ
                    #===================
                    # for NCCL_NET_GDR_READ in 0 1 DEFAULT
                    for NCCL_NET_GDR_READ in DEFAULT
                    do
                        if [ "${NCCL_NET_GDR_READ}" = "DEFAULT" ]
                        then
                            MPIRUN_OPTIONS_GDR_READ=""
                        else
                            MPIRUN_OPTIONS_GDR_READ="-x NCCL_NET_GDR_READ=${NCCL_NET_GDR_READ}"
                        fi

                        echo_hash_line
                        echo "# Test $i..."
                        echo_hash_line

                        echo "INFO: PLUGIN_P2P          = ${PLUGIN_P2P}"
                        echo "INFO: NCCL_PROTO          = ${NCCL_PROTO}"
                        echo "INFO: NCCL_ALGO           = ${NCCL_ALGO}"
                        echo "INFO: SHARP_ENABLE        = ${SHARP_ENABLE}"
                        echo "INFO: NCCL_NET_GDR_LEVEL  = ${NCCL_NET_GDR_LEVEL}"
                        echo "INFO: NCCL_NET_GDR_READ   = ${NCCL_NET_GDR_READ}"

                        CMD="mpirun \
                            ${MPIRUN_OPTIONS_COMMON} \
                            ${MPIRUN_OPTIONS_NCCL_PROTO} \
                            ${MPIRUN_OPTIONS_NCCL_ALGO} \
                            ${MPIRUN_OPTIONS_SHARP} \
                            ${MPIRUN_OPTIONS_GDR_LEVEL} \
                            ${MPIRUN_OPTIONS_GDR_READ} \
                            ${MPIRUN_OPTIONS_PLUGIN_P2P}
                            ${MPI_APP}"
                        echo "# Test $i reproducer:"
                        echo "export PATH=${PATH}"
                        echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
                        trim_multiple_spaces "$CMD"
                        $CMD
                        if [ $? -ne 0 ]
                        then
                            echo "# Test $i... failed"
                            GLOBAL_TEST_STATUS=1
                        else
                            echo "# Test $i... passed"
                        fi

                        i=`expr $i + 1`
                    done
                done
            done
        done
    done
done

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
