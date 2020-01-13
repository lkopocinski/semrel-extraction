#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

INPUT_PATH="./data/corpora"
OUTPUT_PATH="./data/maps"
SCRIPTS_DIR="./data/scripts"

ELMO_PATH="./data/elmo"
FASTTEXT_PATH="./data/fasttext"
RETROFIT_PATH="./data/fasttext"



mkdir -p ${OUTPUT_PATH}

dvc run \
-d ${INPUT_PATH} \
-d ${SCRIPTS_DIR}/make_maps.py \
-o ${OUTPUT_PATH} \
CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/make_maps.py --input-path ${INPUT_PATH} \
                                                   --directories 112 114 115 \
                                                   --elmo-model "${ELMO_PATH}/emb-options.json" "${ELMO_PATH}/emb-weights.hdf5" \
                                                   --fasttext-model "${FASTTEXT_PATH}/kgr10.plain.skipgram.dim300.neg10.bin" \
                                                   --retrofit-model "${RETROFIT_PATH}/kgr10.plain.skipgram.dim300.neg10.retrofit-v3.vec" \
                                                   --output-path ${OUTPUT_PATH} \

popd