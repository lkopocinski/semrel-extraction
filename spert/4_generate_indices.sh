#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

DATA_DIR="./semrel/data/data"
VECTORS_DIR="${DATA_DIR}/vectors"

SCRIPTS_DIR="./spert/scripts"
OUTPUT_DIR="./spert/data"
OUTPUT_PATH="${OUTPUT_DIR}/indices.json"

mkdir -p ${OUTPUT_DIR}

dvc run \
  -d ${DATA_DIR} \
  -d ${VECTORS_DIR} \
  -d ${SCRIPTS_DIR}/generate_indices.py \
  -o ${OUTPUT_PATH} \
  -f _spert.indices.dvc \
  ${SCRIPTS_DIR}/generate_indices.py --dataset-keys ${VECTORS_DIR}/elmo.rel.keys \
                                     --output-path ${OUTPUT_PATH}

popd
