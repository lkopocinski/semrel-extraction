#!/bin/bash -eux

# TODO: description

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./data/generated"
OUTPUT_PATH="./data/sampled"
SCRIPTS_DIR="./data/scripts"

dvc run \
-d ${DATA_IN} \
-d ${SCRIPTS_DIR}/sample_datasets.py \
-o ${OUTPUT_PATH} \
${SCRIPTS_DIR}/sample_datasets.py --data-in ${DATA_IN} \
                                  --output-path ${OUTPUT_PATH} \
                                  --train-size 200 400  \
                                  --valid-size 100 100 \
                                  --test-size 100 100

popd