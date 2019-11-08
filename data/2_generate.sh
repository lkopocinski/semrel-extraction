#!/bin/bash -eux

# Script takes previously split data and generate examples used to train classifier.
# Returns files in context style data format.

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./data/splits"
OUTPUT_PATH="./data/generations"
SCRIPTS_DIR="./data/scripts"

mkdir -p ${OUTPUT_PATH}

dvc run \
-d ${DATA_IN} \
-d ${SCRIPTS_DIR}/generate_examples.py \
-o ${OUTPUT_PATH} \
${SCRIPTS_DIR}/generate_examples.py --data-in ${DATA_IN} \
                                    --output-path ${OUTPUT_PATH}

popd

#sort -u -o ${NEGATIVE_DIR}/${name}.context ${NEGATIVE_DIR}/${name}.context