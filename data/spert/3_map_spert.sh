#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

INDICES_FILE="./data/spert/"
SCRIPTS_DIR="./data/spert"

INPUT_PATH="./data/relations/relations.tsv"
INDICES_FILE="./data/spert/indices.json"
OUTPUT_PATH="./data/spert/spert.json"

dvc run \
  -d ${INPUT_PATH} \
  -d ${SCRIPTS_DIR}/relations_to_spert_json.py \
  -o ${OUTPUT_PATH} \
  -f spert_map.dvc
  CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/relations_to_spert_json.py --input-path ${INPUT_PATH} \
                                                                   --indices-file ${INDICES_FILE} \
                                                                   --output-path ${OUTPUT_PATH}

popd