#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

SCRIPTS_DIR="./data/scripts"
MAPS_DIR="./data/maps"

INPUT_PATH="./data/relations/relations.tsv"
OUTPUT_DIR="./data/vectors"

mkdir -p ${OUTPUT_DIR}

dvc run \
  -d ${INPUT_PATH} \
  -d ${SCRIPTS_DIR}/embedd_relations.py \
  -d ${MAPS_DIR} \
  -o ${OUTPUT_DIR} \
  CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/embedd_relations.py --input-path ${INPUT_PATH} \
                                                            --elmo-map "${MAPS_DIR}/elmo.map.keys" "${MAPS_DIR}/elmo.map.pt" \
                                                            --fasttext-map "${MAPS_DIR}/fasttext.map.keys" "${MAPS_DIR}/fasttext.map.pt" \
                                                            --retrofit-map "${MAPS_DIR}/retrofit.map.keys" "${MAPS_DIR}/retrofit.map.pt" \
                                                            --output-dir ${OUTPUT_DIR}

popd
