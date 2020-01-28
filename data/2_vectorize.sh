#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./data/relations"
OUTPUT_PATH="./data/vectors"
SCRIPTS_DIR="./data/scripts"
MAPS_DIR="./data/maps"

dvc run \
-d ${DATA_IN} \
-d ${MAPS_DIR} \
-d ${SCRIPTS_DIR}/combine_vectors.py \
-o ${OUTPUT_PATH} \

${SCRIPTS_DIR}/combine_vectors.py --data-in "${DATA_IN}/relations." \
                                 --output-path ${OUTPUT_PATH} \
                                 --elmo-map "${MAPS_DIR}/elmo.map.pt" \
                                 --fasttext-map "${MAPS_DIR}/fasttext.map.pt" \
                                 --retrofit-map "${MAPS_DIR}/retrofit.map.pt"

popd