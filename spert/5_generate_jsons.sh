#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

DATA_DIR="./semrel/data/data"

SCRIPTS_DIR="./spert/scripts"
INDICES_FILE="./spert/data/indices.json"

INPUT_PATH="${DATA_DIR}/relations/relations.tsv"
OUTPUT_DIR="./spert/data/dataset"

mkdir -p ${OUTPUT_DIR}

dvc run \
  -d ${INPUT_PATH} \
  -d ${INDICES_FILE} \
  -d ${SCRIPTS_DIR}/generate_spert_json.py \
  -o ${OUTPUT_DIR} \
  -f _spert.jsons.dvc \
  ${SCRIPTS_DIR}/generate_spert_json.py --input-path ${INPUT_PATH} \
                                        --indices-file ${INDICES_FILE} \
                                        --output-dir ${OUTPUT_DIR}

popd
