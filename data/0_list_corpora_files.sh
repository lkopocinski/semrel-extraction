#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

CORPORA_DIR="./data/corpora/"
OUTPUT_FILE="./data/relations_files.list"

dvc run \
  -d ${CORPORA_DIR} \
  -o ${OUTPUT_FILE} \
  "find ${CORPORA_DIR}117/ -type f -name "*.rel.xml" > ${OUTPUT_FILE}"

popd
