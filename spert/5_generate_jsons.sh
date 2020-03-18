#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

SCRIPTS_DIR="./data/spert"
INDICES_FILE="./data/spert/indices.json"

INPUT_PATH="./data/relations/relations.tsv"
OUTPUT_DIR="./data/spert/dataset"

mkdir -p ${OUTPUT_DIR}

dvc run \
  -d ${INPUT_PATH} \
  -d ${INDICES_FILE} \
  -d ${SCRIPTS_DIR}/generate_spert_json.py \
  -o ${OUTPUT_DIR} \
  -f spert.jsons.dvc \
  ${SCRIPTS_DIR}/generate_spert_json.py --input-path ${INPUT_PATH} \
                                        --indices-file ${INDICES_FILE} \
                                        --output-dir ${OUTPUT_DIR}

popd
