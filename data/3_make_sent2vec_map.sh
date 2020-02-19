#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

RELATIONS_FILE="./data/relations/relations.tsv"
DOCUMENTS_FILE="./data/relations_files.txt"
MODEL_DIR="./data/sent2vec"

OUTPUT_DIR="./data/vectors"
SCRIPTS_DIR="./data/scripts"

dvc run \
  -d ${RELATIONS_FILE} \
  -d ${DOCUMENTS_FILE} \
  -d ${SCRIPTS_DIR}/make_sent2vec_map.py \
  -o ${OUTPUT_DIR} \
  CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/make_sent2vec_map.py --relations-file ${RELATIONS_FILE} \
                                                             --documents-files ${DOCUMENTS_FILE} \
                                                             --model "${MODEL_DIR}/kgr10.bin" \
                                                             --output-paths "${OUTPUT_DIR}/sent2vec.map.keys" "${OUTPUT_DIR}/sent2vec.map.pt"

popd