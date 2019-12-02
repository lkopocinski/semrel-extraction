#!/bin/bash

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./data/generations"
OUTPUT_PATH="./data/vectorss"
SCRIPTS_DIR="./data/scripts"

#${SCRIPTS_DIR}/combine_vectors.py --data-in ${DATA_IN} \
#                                 --output-path ${OUTPUT_PATH} \
#                                 --elmo-map "./data/maps/1xx.corpus.elmo.map" \
#                                 --fasttext-map "./data/maps/1xx.corpus.fasttext.map" \
#                                 --retrofit-map "./data/maps/1xx.corpus.retrofit.map"

${SCRIPTS_DIR}/draft/vectors_map_sent2vec.py --relations-file ${DATA_IN}/relations.112.114.115.context.uniq \
                                  --sent2vec "./data/sent2vec/kgr10.bin"

popd
