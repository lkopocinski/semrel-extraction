#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

CORPORA_DIR="./data/corpora/"
OUTPUT_PATH="./data/relations_files.txt"

dvc run \
  -d ${CORPORA_DIR} \
  -o ${OUTPUT_PATH} \
  "find ${CORPORA_DIR}116/ -type f -name "*.rel.xml" > ${OUTPUT_PATH}"

popd
