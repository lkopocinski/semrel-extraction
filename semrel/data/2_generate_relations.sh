#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

SCRIPTS_DIR="./semrel/data/scripts/cli"
OUTPUT_DIR="./semrel/data/data/relations"

INPUT_PATH="./semrel/data/data/relations_files.list"
RELATIONS_PATH="${OUTPUT_DIR}/relations.tsv"

mkdir -p ${OUTPUT_DIR}

dvc run \
  -d ${INPUT_PATH} \
  -d ${SCRIPTS_DIR}/generate_relations.py \
  -d ${SCRIPTS_DIR}/relations.py \
  -o ${RELATIONS_PATH} \
  CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/generate_relations.py --input-path ${INPUT_PATH} \
                                                              --output-path ${RELATIONS_PATH}

popd
