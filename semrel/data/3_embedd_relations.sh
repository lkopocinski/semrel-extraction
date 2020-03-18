#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

ROOT_DIR="./semrel/data"
DATA_DIR="${ROOT_DIR}/data"

SCRIPTS_DIR="${ROOT_DIR}/scripts"
CLI_DIR="${SCRIPTS_DIR}/cli"
MAPS_DIR="${DATA_DIR}/maps"

INPUT_PATH="${DATA_DIR}/relations/relations.tsv"
OUTPUT_DIR="${DATA_DIR}/vectors"

mkdir -p ${OUTPUT_DIR}

dvc run \
  -d ${INPUT_PATH} \
  -d ${CLI_DIR}/embedd_relations.py \
  -d ${MAPS_DIR} \
  -o ${OUTPUT_DIR}/elmo.rel.keys \
  -o ${OUTPUT_DIR}/elmo.rel.pt \
  -o ${OUTPUT_DIR}/fasttext.rel.keys \
  -o ${OUTPUT_DIR}/fasttext.rel.pt \
  -o ${OUTPUT_DIR}/retrofit.rel.keys \
  -o ${OUTPUT_DIR}/retrofit.rel.pt \
  -f _vectors.dvc
  CUDA_VISIBLE_DEVICES=0 ${CLI_DIR}/embedd_relations.py --input-path ${INPUT_PATH} \
                                                        --elmo-map "${MAPS_DIR}/elmo.map.keys" "${MAPS_DIR}/elmo.map.pt" \
                                                        --fasttext-map "${MAPS_DIR}/fasttext.map.keys" "${MAPS_DIR}/fasttext.map.pt" \
                                                        --retrofit-map "${MAPS_DIR}/retrofit.map.keys" "${MAPS_DIR}/retrofit.map.pt" \
                                                        --output-dir ${OUTPUT_DIR}

popd
