#!/bin/bash -eE
# Preparation a workplace & configs to CI
function configure() {
    rm -rf "${NFS_WORKSPACE}-pr" || true
    rm -rf "${NFS_WORKSPACE}" || true
    rm -rf ./nccl-rdma-sharp-plugins/.ci/cfg/* || true
    cd "${NFS_WORKSPACE_ROOT}" || exit 1
    mkdir -p ./nccl-rdma-sharp-plugins/.ci/cfg/"${HOST1}"/sharp_conf 
    mkdir -p ./nccl-rdma-sharp-plugins/.ci/cfg/"${HOST2}"/sharp_conf

    printf "%s\n%s\n" "${HOST1}" "${HOST2}" >./nccl-rdma-sharp-plugins/.ci/cfg/"${HOST1}"/hostfile
    printf "log_verbosity 3\n" >./nccl-rdma-sharp-plugins/.ci/cfg/"${HOST1}"/sharp_conf/sharp_am.cfg
    printf "log_verbosity 3\n" >./nccl-rdma-sharp-plugins/.ci/cfg/"${HOST1}"/sharp_conf/sharpd.cfg
    printf "%s\n" "${SHARP_AM_HOST}" >./nccl-rdma-sharp-plugins/.ci/cfg/"${HOST1}"/sharp_conf/sharp_am_node.txt

    printf "%s\n%s\n" "${HOST1}" "${HOST2}" >./nccl-rdma-sharp-plugins/.ci/cfg/"${HOST2}"/hostfile
    printf "log_verbosity 3\n" >./nccl-rdma-sharp-plugins/.ci/cfg/"${HOST2}"/sharp_conf/sharp_am.cfg
    printf "log_verbosity 3\n" >./nccl-rdma-sharp-plugins/.ci/cfg/"${HOST2}"/sharp_conf/sharpd.cfg
    printf "%s\n" "${SHARP_AM_HOST}" >./nccl-rdma-sharp-plugins/.ci/cfg/"${HOST2}"/sharp_conf/sharp_am_node.txt
}

# Building NCCL rdma sharp plugin
function build() {
    echo "Running build_nccl_rdma_sharp_plugins.sh..."
    "${WORKSPACE}"/.ci/build_nccl_rdma_sharp_plugins.sh && echo "Build SUCCESFULL !!!"
}

# Checking and configuring Sharp
function sharp() {
    echo "Running configure_sharp.sh..."
    "${WORKSPACE}"/.ci/configure_sharp.sh && echo "Step configure_sharp SUCCESFULL !!!"
}

# Running of tests
function test() {
    git clone --depth=1 https://github.com/NVIDIA/nccl-tests.git "${NFS_WORKSPACE}"/nccl-tests
    echo "Running run_nccl_tests.sh..."
    "${WORKSPACE}"/.ci/run_nccl_tests.sh && echo "Tests SUCCESFULL !!!"
}
