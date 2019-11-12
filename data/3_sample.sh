#!/bin/bash -eux

# Script randomly selects elements according to the given amount of set.

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./data/generations"
OUTPUT_PATH="./data/sampled"
SCRIPTS_DIR="./data/scripts"

mkdir -p ${OUTPUT_PATH}

dvc run \
-d ${DATA_IN} \
-d ${SCRIPTS_DIR}/sample.py \
-o ${OUTPUT_PATH} \
${SCRIPTS_DIR}/sample.py --data-in ${DATA_IN} \
                        --output-path ${OUTPUT_PATH} \
                        --train-size 200 400  \
                        --valid-size 100 100 \
                        --test-size 100 100

popd
