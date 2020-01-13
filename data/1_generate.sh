#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

INPUT_PATH="./data/corpora"
OUTPUT_PATH="./data/relations"
SCRIPTS_DIR="./data/scripts"

mkdir -p ${OUTPUT_PATH}

dvc run \
-d ${INPUT_PATH} \
-d ${SCRIPTS_DIR}/generate.py \
-d ${SCRIPTS_DIR}/generator.py \
-o ${OUTPUT_PATH} \
CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/generate.py --input-path ${INPUT_PATH} \
                                                  --directories 112 114 115 \
                                                  --output-path "${OUTPUT_PATH}/relations.tsv" \

popd