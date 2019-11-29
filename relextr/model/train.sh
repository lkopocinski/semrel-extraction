#!/bin/bash -eux

# Script runs model training session

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./relextr/model/dataset"
SCRIPTS_DIR="./relextr/model/scripts"
CONFIG="./relextr/model/config.yaml"

dvc run \
-d ${DATA_IN} \
-d ${SCRIPTS_DIR}/train.py \
-d ${SCRIPTS_DIR}/relnet.py \
-d ${SCRIPTS_DIR}/utils/utils.py \
-d ${SCRIPTS_DIR}/utils/metrics.py \
-d ${SCRIPTS_DIR}/utils/batches.py \
-d ${CONFIG} \
-M metrics.txt \
-f train.dvc \
CUDA_VISIBLE_DEVICES=1,2,3 ${SCRIPTS_DIR}/train.py --data-in ${DATA_IN} \
                                                      --config ${CONFIG}

popd
