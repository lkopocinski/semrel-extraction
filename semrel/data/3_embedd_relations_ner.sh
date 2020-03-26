#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

ROOT_DIR="./semrel/data"
DATA_DIR="${ROOT_DIR}/data"

RELATIONS_FILE="${DATA_DIR}/relations/relations.tsv"

SCRIPTS_DIR="${ROOT_DIR}/scripts"
CLI_DIR="${SCRIPTS_DIR}/cli"
OUTPUT_DIR="${DATA_DIR}/vectors"

KEYS_FILE="${OUTPUT_DIR}/ner.rel.keys"
VECTORS_FILE="${OUTPUT_DIR}/ner.rel.pt"

mkdir -p ${OUTPUT_DIR}

dvc run \
  -d ${RELATIONS_FILE} \
  -d ${CLI_DIR}/make_ner_map.py \
  -o ${KEYS_FILE} \
  -o ${VECTORS_FILE} \
  -f _vectors.ner.dvc \
  ${CLI_DIR}/make_ner_map.py --relations-file ${RELATIONS_FILE} \
                             --output-paths ${KEYS_FILE} ${VECTORS_FILE}

popd
