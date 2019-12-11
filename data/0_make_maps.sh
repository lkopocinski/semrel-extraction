#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./data/corpora"
OUTPUT_PATH="./data/maps"
SCRIPTS_DIR="./data/scripts"

mkdir -p ${OUTPUT_PATH}

dvc run \
-d ${DATA_IN} \
-d ${SCRIPTS_DIR}/make_maps.py \
-o ${OUTPUT_PATH} \
${SCRIPTS_DIR}/make_maps --data-in ${DATA_IN} \
                           --output-path ${OUTPUT_PATH} \
                           --directories 112 114 115

popd