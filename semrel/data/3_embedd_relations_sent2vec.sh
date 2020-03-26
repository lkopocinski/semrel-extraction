#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

ROOT_DIR="./semrel/data"
DATA_DIR="${ROOT_DIR}/data"

RELATIONS_FILE="${DATA_DIR}/relations/relations.tsv"
DOCUMENTS_FILE="${DATA_DIR}/relations.files.list"
MODEL_DIR="${DATA_DIR}/sent2vec"

SCRIPTS_DIR="${ROOT_DIR}/scripts"
CLI_DIR="${SCRIPTS_DIR}/cli"
OUTPUT_DIR="${DATA_DIR}/vectors"

KEYS_FILE="${OUTPUT_DIR}/sent2vec.rel.keys"
VECTORS_FILE="${OUTPUT_DIR}/sent2vec.rel.pt"

mkdir -p ${OUTPUT_DIR}

dvc run \
  -d ${RELATIONS_FILE} \
  -d ${DOCUMENTS_FILE} \
  -d ${CLI_DIR}/make_sent2vec_map.py \
  -o ${KEYS_FILE} \
  -o ${VECTORS_FILE} \
  -f _vectors.sent2vec.dvc \
  CUDA_VISIBLE_DEVICES=0 ${CLI_DIR}/make_sent2vec_map.py --relations-file ${RELATIONS_FILE} \
                                                         --documents-files ${DOCUMENTS_FILE} \
                                                         --model "${MODEL_DIR}/kgr10.bin" \
                                                         --output-paths ${KEYS_FILE} ${VECTORS_FILE}

popd
