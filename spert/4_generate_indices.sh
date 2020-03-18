#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

DATA_DIR="./data/vectors"
SCRIPTS_DIR="./data/spert"
OUTPUT_PATH="./data/spert/indices.json"

dvc run \
  -d ${DATA_DIR} \
  -d ${SCRIPTS_DIR}/generate_indices.py \
  -O ${OUTPUT_PATH} \
  -f spert.indices.dvc \
  ${SCRIPTS_DIR}/generate_indices.py --dataset-keys ${DATA_DIR}/elmo.rel.keys \
                                     --output-path ${OUTPUT_PATH}

popd
