#!/bin/bash -eux

# TODO: description

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./data/sampled"
OUTPUT_PATH="./data/vectors"
SCRIPTS_DIR="./data/scripts"

dvc run \
-d ${DATA_IN} \
-d ${SCRIPTS_DIR}/create_vectors.py \
-o ${OUTPUT_PATH} \
${SCRIPTS_DIR}/create_vectors.py --data-in ${DATA_IN} \
                                 --output-path ${OUTPUT_PATH} \
                                 --options "./data/elmo/emb-options.json"
                                 --weights "./data/elmo/emb-weights.hdf5"

popd