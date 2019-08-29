#!/bin/bash -leE

if [ -n "$DEBUG" ]
then
    set -x
fi

echo 'Publish artefacts...'

module load ml/ci-tools

export UPSTREAM_JOB_NAME=${JOB_NAME}
export UPSTREAM_BUILD_NUMBER=${BUILD_NUMBER}
export UPSTREAM_ghprbGhRepository=${ghprbGhRepository}
export UPSTREAM_ghprbPullId=${ghprbPullId}

ls -al ${ARTEFACT_DIR}

publish_artefacts_to_gist.py

echo 'Publish artefacts... DONE'
