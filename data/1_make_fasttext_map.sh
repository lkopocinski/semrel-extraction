#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

MODEL_DIR="./data/fasttext"
SCRIPTS_DIR="./data/scripts"

INPUT_PATH="./data/relations_files.list"
KEYS_FILE="./data/maps/fasttext.map.keys"
VECTORS_FILE="./data/maps/fasttext.map.pt"

mkdir -p "./data/maps/"

dvc run \
  -d ${INPUT_PATH} \
  -d ${SCRIPTS_DIR}/make_fasttext_map.py \
  -d ${SCRIPTS_DIR}/maps.py \
  -o ${KEYS_FILE} \
  -o ${VECTORS_FILE} \
  -f fasttext.map.dvc \
  CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/make_fasttext_map.py --input-path ${INPUT_PATH} \
                                                             --model "${MODEL_DIR}/kgr10.plain.skipgram.dim300.neg10.bin" \
                                                             --output-paths ${KEYS_FILE} ${VECTORS_FILE}

popd
