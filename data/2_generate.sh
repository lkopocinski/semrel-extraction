#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

SCRIPTS_DIR="./data/scripts"
OUTPUT_DIR="./data/relations"

INPUT_PATH="./data/relations_files.list"
RELATIONS_PATH="${OUTPUT_DIR}/relations.tsv"

mkdir -p ${OUTPUT_DIR}

dvc run \
  -d ${INPUT_PATH} \
  -d ${SCRIPTS_DIR}/generate_relations.py \
  -d ${SCRIPTS_DIR}/generator.py \
  -o ${RELATIONS_PATH} \
  CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/generate_relations.py --input-path ${INPUT_PATH} \
                                                              --output-path ${RELATIONS_PATH}

popd
