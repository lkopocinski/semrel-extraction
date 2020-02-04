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
CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/combine_vectors.py --data-in "${INPUT_PATH}/relations.tsv" \
                                                         --elmo-map "${MAPS_DIR}/elmo.map.pt" \
                                                         --fasttext-map "${MAPS_DIR}/fasttext.map.pt" \
                                                         --retrofit-map "${MAPS_DIR}/retrofit.map.pt" \
                                                         --output-path ${OUTPUT_PATH}

popd