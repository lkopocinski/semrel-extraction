#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

MODEL_DIR="./data/fasttext"
SCRIPTS_DIR="./data/scripts"

INPUT_PATH="./data/relations_files.txt"
KEYS_FILE="./data/maps/retrofit.map.keys"
VECTORS_FILE="./data/maps/retrofit.map.pt"

mkdir -p "./data/maps/"

dvc run \
  -d ${INPUT_PATH} \
  -d ${SCRIPTS_DIR}/make_retrofit_map.py \
  -d ${SCRIPTS_DIR}/maps.py \
  -o ${KEYS_FILE} \
  -o ${VECTORS_FILE} \
  -f retrofit.map.dvc \
  CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/make_retrofit_map.py --input-path ${INPUT_PATH} \
                                                             --model-retrofit "${MODEL_DIR}/kgr10.plain.skipgram.dim300.neg10.retrofit.vec" \
                                                             --model-fasttext "${MODEL_DIR}/kgr10.plain.skipgram.dim300.neg10.bin" \
                                                             --output-paths ${KEYS_FILE} ${VECTORS_FILE}

popd
