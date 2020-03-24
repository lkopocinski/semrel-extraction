#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

DATA_DIR="./semrel/data/data"
KEYS_PATH="${DATA_DIR}/vectors/elmo.rel.keys"

SCRIPTS_DIR="./spert/scripts"
OUTPUT_DIR="./spert/data"
OUTPUT_PATH="${OUTPUT_DIR}/indices.json"

mkdir -p ${OUTPUT_DIR}

dvc run \
  -d ${KEYS_PATH} \
  -d ${SCRIPTS_DIR}/generate_indices.py \
  -o ${OUTPUT_PATH} \
  -f _spert.indices.dvc \
  ${SCRIPTS_DIR}/generate_indices.py --dataset-keys ${KEYS_PATH} \
                                     --output-path ${OUTPUT_PATH}

popd
