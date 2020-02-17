#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

INPUT_PATH="./data/relations_files.txt"
OUTPUT_DIR="./data/relations"
OUTPUT_PATH="${OUTPUT_DIR}/relations.tsv"
SCRIPTS_DIR="./data/scripts"

mkdir -p ${OUTPUT_DIR}

dvc run \
  -d ${INPUT_PATH} \
  -d ${SCRIPTS_DIR}/generate.py \
  -d ${SCRIPTS_DIR}/generator.py \
  -o ${OUTPUT_PATH} \
  CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/generate.py --input-path ${INPUT_PATH} \
                                                    --output-path ${OUTPUT_PATH}

popd
