#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

ROOT_DIR="./semrel/data"
DATA_DIR="${ROOT_DIR}/data"

MODEL_DIR="${DATA_DIR}/fasttext"
SCRIPTS_DIR="${ROOT_DIR}/scripts"
CLI_DIR="${SCRIPTS_DIR}/cli"

INPUT_PATH="${DATA_DIR}/relations.files.list"
KEYS_FILE="${DATA_DIR}/maps/retrofit.map.keys"
VECTORS_FILE="${DATA_DIR}/maps/retrofit.map.pt"

mkdir -p "${DATA_DIR}/maps/"

dvc run \
  -d ${INPUT_PATH} \
  -d ${CLI_DIR}/make_retrofit_map.py \
  -d ${SCRIPTS_DIR}/maps.py \
  -o ${KEYS_FILE} \
  -o ${VECTORS_FILE} \
  -f _retrofit.map.dvc \
  CUDA_VISIBLE_DEVICES=0 ${CLI_DIR}/make_retrofit_map.py --input-path ${INPUT_PATH} \
                                                         --model-retrofit "${MODEL_DIR}/kgr10.plain.skipgram.dim300.neg10.retrofit.vec" \
                                                         --model-fasttext "${MODEL_DIR}/kgr10.plain.skipgram.dim300.neg10.bin" \
                                                         --output-paths ${KEYS_FILE} ${VECTORS_FILE}

popd
