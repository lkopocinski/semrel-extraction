#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

ROOT_DIR="./semrel/data"
DATA_DIR="${ROOT_DIR}/data"

MODEL_DIR="${DATA_DIR}/elmo"
SCRIPTS_DIR="${ROOT_DIR}/scripts"
CLI_DIR="${SCRIPTS_DIR}/cli"

INPUT_PATH="${DATA_DIR}/relations.files.list"
KEYS_FILE="${DATA_DIR}/maps/elmo.map.keys"
VECTORS_FILE="${DATA_DIR}/maps/elmo.map.pt"

mkdir -p "${DATA_DIR}/maps/"

dvc run \
  -d ${INPUT_PATH} \
  -d ${CLI_DIR}/make_elmo_map.py \
  -d ${SCRIPTS_DIR}/maps.py \
  -o ${KEYS_FILE} \
  -o ${VECTORS_FILE} \
  -f _elmo.map.dvc \
  CUDA_VISIBLE_DEVICES=0 ${CLI_DIR}/make_elmo_map.py --input-path ${INPUT_PATH} \
                                                     --model "${MODEL_DIR}/options.json" "${MODEL_DIR}/weights.hdf5" \
                                                     --output-paths ${KEYS_FILE} ${VECTORS_FILE}

popd
