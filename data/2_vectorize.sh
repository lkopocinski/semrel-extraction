#!/bin/bash

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./data/generations"
OUTPUT_PATH="./data/vectors"
SCRIPTS_DIR="./data/scripts"

mkdir -p ${OUTPUT_PATH}

dvc run \
-d ${DATA_IN} \
-d ${SCRIPTS_DIR}/combine_vectors.py \
-o ${OUTPUT_PATH} \
${SCRIPTS_DIR}/combine_vectors.py --data-in ${DATA_IN} \
                                 --output-path ${OUTPUT_PATH} \
                                 --elmo-map "./data/maps/elmo.fake.map" \
                                 --fasttext-map "./data/maps/fasttext.fake.map" \
                                 --retrofit-map "./data/maps/retrofit.fake.map" 

popd
