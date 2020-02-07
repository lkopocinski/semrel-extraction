#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

INPUT_PATH="./data/relations"
OUTPUT_PATH="./data/vectors"
SCRIPTS_DIR="./data/scripts"
MAPS_DIR="./data/maps"

dvc run \
-d ${INPUT_PATH} \
-d ${MAPS_DIR} \
-d ${SCRIPTS_DIR}/combine_vectors.py \
-o ${OUTPUT_PATH} \
CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/combine_vectors.py --input-path "${INPUT_PATH}/relations.tsv" \
                                                         --elmo-map "${MAPS_DIR}/elmo.map.pt" "${MAPS_DIR}/elmo.map.keys" \
                                                         --fasttext-map "${MAPS_DIR}/fasttext.map.pt" "${MAPS_DIR}/fasttext.map.keys" \
                                                         --retrofit-map "${MAPS_DIR}/retrofit.map.pt" "${MAPS_DIR}/retrofit.map.keys" \
                                                         --output-path ${OUTPUT_PATH}

popd