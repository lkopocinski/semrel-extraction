#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

RELATIONS_FILE="./data/relations/relations.tsv"
DOCUMENTS_FILE="./data/relations_files.list"
MODEL_DIR="./data/sent2vec"

SCRIPTS_DIR="./data/scripts"
OUTPUT_DIR='./data/vectors'

KEYS_FILE="${OUTPUT_DIR}/sent2vec.map.keys"
VECTORS_FILE="${OUTPUT_DIR}/sent2vec.map.pt"

mkdir -p ${OUTPUT_DIR}

dvc run \
  -d ${RELATIONS_FILE} \
  -d ${DOCUMENTS_FILE} \
  -d ${SCRIPTS_DIR}/make_sent2vec_map.py \
  -o ${KEYS_FILE} \
  -o ${VECTORS_FILE} \
  CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/make_sent2vec_map.py --relations-file ${RELATIONS_FILE} \
                                                             --documents-files ${DOCUMENTS_FILE} \
                                                             --model "${MODEL_DIR}/kgr10.bin" \
                                                             --output-paths ${KEYS_FILE} ${VECTORS_FILE}

popd