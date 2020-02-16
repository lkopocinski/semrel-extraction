#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

MODEL_DIR="./data/elmo"
SCRIPTS_DIR="./data/scripts"

INPUT_PATH="./data/relations_files.txt"
KEYS_FILE="./data/maps/elmo.map.keys"
VECTORS_FILE="./data/maps/elmo.map.pt"

dvc run \
  -d ${INPUT_PATH} \
  -d ${SCRIPTS_DIR}/make_elmo_map.py \
  -o ${KEYS_FILE} \
  -o ${VECTORS_FILE} \
  -f elmo.map.dvc
  CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/make_elmo_map.py --input-path ${INPUT_PATH} \
                                                         --model "${MODEL_DIR}/options.json" "${MODEL_DIR}/weights.hdf5" \
                                                         --output-path ${KEYS_FILE} ${VECTORS_FILE}

popd
