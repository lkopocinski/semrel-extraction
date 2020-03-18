#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

RELATIONS_FILE="./semrel/data/data/relations/relations.tsv"
DOCUMENTS_FILE="./semrel/data/data/relations.files.list"
MODEL_DIR="./semrel/data/data/sent2vec"

SCRIPTS_DIR="./semrel/data/data/scripts/cli"
OUTPUT_DIR='./semrel/data/data/vectors'

KEYS_FILE="${OUTPUT_DIR}/sent2vec.rel.keys"
VECTORS_FILE="${OUTPUT_DIR}/sent2vec.rel.pt"

mkdir -p ${OUTPUT_DIR}

dvc run \
  -d ${RELATIONS_FILE} \
  -d ${DOCUMENTS_FILE} \
  -d ${SCRIPTS_DIR}/make_sent2vec_map.py \
  -o ${KEYS_FILE} \
  -o ${VECTORS_FILE} \
  -f vectors.sent2vec.dvc \
  CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/make_sent2vec_map.py --relations-file ${RELATIONS_FILE} \
                                                             --documents-files ${DOCUMENTS_FILE} \
                                                             --model "${MODEL_DIR}/kgr10.bin" \
                                                             --output-paths ${KEYS_FILE} ${VECTORS_FILE}

popd
