#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./data/relations"
OUTPUT_PATH="./data/vectors"
SCRIPTS_DIR="./data/scripts"
MAPS_DIR="./data/maps"

dvc run \
-d ${DATA_IN} \
-d ${MAPS_DIR} \
-o ${OUTPUT_PATH} \

${SCRIPTS_DIR}/combine_vectors.py --data-in "${DATA_IN}/relations." \
                                 --output-path ${OUTPUT_PATH} \
                                 --elmo-map "${MAPS_DIR}/elmo.map.pt" \
                                 --fasttext-map "${MAPS_DIR}/fasttext.map.pt" \
                                 --retrofit-map "${MAPS_DIR}/retrofit.map.pt"

popd

#${SCRIPTS_DIR}/draft/vectors_map_sent2vec.py --relations-file ${DATA_IN}/relations.112.114.115.context.uniq \
#  --sent2vec "./data/sent2vec/kgr10.bin"