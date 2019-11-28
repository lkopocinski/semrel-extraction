#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./data/generations"
OUTPUT_PATH="./data/vectors"
SCRIPTS_DIR="./data/scripts"

dvc run \
-d ${DATA_IN} \
-d ${SCRIPTS_DIR}/combine_vectors.py \
-o ${OUTPUT_PATH} \
${SCRIPTS_DIR}/combine_vectors.py --data-in ${DATA_IN} \
                                 --output-path ${OUTPUT_PATH} \
                                 --elmo-map "./data/maps/1xx.corpus.elmo.map" \
                                 --fasttext-map "./data/maps/1xx.corpus.fasttext.map" \
                                 --retrofit-model "./data/maps/1xx.corpus.retrofit.map" \

#                                 --sent2vec-map "./data/maps/1xx.corpus.sent2vec.map" \
#                                 --elmoconv-map "./data/maps/1xx.corpus.elmoconv.map" \

popd