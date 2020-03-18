#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

RELATIONS_FILE="./semrel/data/data/relations/relations.tsv"

SCRIPTS_DIR="./semrel/data/data/scripts/cli"
OUTPUT_DIR="./semrel/data/data/vectors"

KEYS_FILE="${OUTPUT_DIR}/ner.rel.keys"
VECTORS_FILE="${OUTPUT_DIR}/ner.rel.pt"

mkdir -p ${OUTPUT_DIR}

dvc run \
  -d ${RELATIONS_FILE} \
  -d ${SCRIPTS_DIR}/make_ner_map.py \
  -o ${KEYS_FILE} \
  -o ${VECTORS_FILE} \
  -f vectors.ner.dvc \
  ${SCRIPTS_DIR}/make_ner_map.py --relations-file ${RELATIONS_FILE} \
                                 --output-paths ${KEYS_FILE} ${VECTORS_FILE}

popd
