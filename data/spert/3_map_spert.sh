#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

INPUT_PATH="./data/relations/relations.tsv"
OUTPUT_PATH="./data/spert/spert.json"
SCRIPTS_DIR="./data/spert"

dvc run \
  -d ${INPUT_PATH} \
  -d ${SCRIPTS_DIR}/relations_to_spert_json.py \
  -o ${OUTPUT_PATH} \
  CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/relations_to_spert_json.py --input-path ${INPUT_PATH} \
                                                                   --output-path ${OUTPUT_PATH}

popd