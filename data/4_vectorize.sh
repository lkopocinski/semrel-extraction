#!/bin/bash -eux

# Script creates context vectors for the elements of each data set

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./data/sampled"
OUTPUT_PATH="./data/vectors"
SCRIPTS_DIR="./data/scripts"

dvc run \
-d ${DATA_IN} \
-d ${SCRIPTS_DIR}/vectorize.py \
-o ${OUTPUT_PATH} \
${SCRIPTS_DIR}/vectorize.py --data-in ${DATA_IN} \
                                 --output-path ${OUTPUT_PATH} \
                                 --options "./data/elmo/emb-options.json" \
                                 --weights "./data/elmo/emb-weights.hdf5"

popd