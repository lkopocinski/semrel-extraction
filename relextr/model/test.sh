#!/bin/bash -eux

# Script runs test training session

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./relextr/model/dataset"
SCRIPTS_DIR="./relextr/model/scripts"
CONFIG="./relextr/model/config/test.yaml"

dvc run \
-d ${DATA_IN} \
-d ${SCRIPTS_DIR}/test.py \
-d ${SCRIPTS_DIR}/relnet.py \
-d ${SCRIPTS_DIR}/utils/engines.py \
-d ${SCRIPTS_DIR}/utils/utils.py \
-d ${SCRIPTS_DIR}/utils/metrics.py \
-d ${SCRIPTS_DIR}/utils/batches.py \
-d ${CONFIG} \
-M metrics_test.txt \
-f test_oneout.dvc \
CUDA_VISIBLE_DEVICES=7,8,9,10 ${SCRIPTS_DIR}/test.py --data-in ${DATA_IN} \
                                                     --config ${CONFIG}

popd
