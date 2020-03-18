#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

MODEL_DIR="./semrel/data/data/fasttext"
SCRIPTS_DIR="./semrel/data/scripts/cli"

INPUT_PATH="./semrel/data/data/relations.files.list"
KEYS_FILE="./semrel/data/data/maps/retrofit.map.keys"
VECTORS_FILE="./semrel/data/data/maps/retrofit.map.pt"

mkdir -p "./semrel/data/data/maps/"

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
