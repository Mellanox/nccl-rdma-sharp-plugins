#!/bin/bash -eE
. ./pushd_functions.sh
. ./ci_functions.sh
pushd /GIT
case $1 in
    build)
        configure
        echo "Building NCCL sharp plugin"
        build
        ;;
    sharp)
        echo "Checking and configure sharp"
        sharp
        ;;
    test)
        echo "Running tests for NCCL sharp plugin"
        test
        ;;
    *)
        echo "Do nothing"
        ;;
esac
