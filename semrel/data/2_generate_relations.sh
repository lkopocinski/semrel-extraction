#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

ROOT_DIR="./semrel/data"
DATA_DIR="${ROOT_DIR}/data"

SCRIPTS_DIR="${ROOT_DIR}/scripts"
CLI_DIR="${SCRIPTS_DIR}/cli"
OUTPUT_DIR="${DATA_DIR}/relations"

INPUT_PATH="${DATA_DIR}/relations.files.list"
RELATIONS_PATH="${OUTPUT_DIR}/relations.tsv"

mkdir -p ${OUTPUT_DIR}

dvc run \
  -d ${INPUT_PATH} \
  -d ${CLI_DIR}/generate_relations.py \
  -d ${SCRIPTS_DIR}/relations.py \
  -o ${RELATIONS_PATH} \
  -f _relations.tsv.dvc
  CUDA_VISIBLE_DEVICES=0 ${CLI_DIR}/generate_relations.py --input-path ${INPUT_PATH} \
                                                          --output-path ${RELATIONS_PATH}

popd
