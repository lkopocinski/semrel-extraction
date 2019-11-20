#!/bin/bash -eux

# Script takes files from provided corpora and splits them into train, valid, test
# disjoint data sets lists. It means each one of the splits contains different files.

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./data/corpora"
OUTPUT_PATH="./data/splits"
SCRIPTS_DIR="./data/scripts"

mkdir -p ${OUTPUT_PATH}

dvc run \
-d ${DATA_IN} \
-d ${SCRIPTS_DIR}/split.py \
-o ${OUTPUT_PATH} \
${SCRIPTS_DIR}/split.py --data-in ${DATA_IN} \
                        --output-path ${OUTPUT_PATH} \
                        --directories 81 82 83

popd
