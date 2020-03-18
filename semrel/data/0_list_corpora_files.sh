#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

CORPORA_DIR="./semrel/data/data/corpora/"
OUTPUT_FILE="./semrel/data/data/relations.files.list"

dvc run \
  -d ${CORPORA_DIR} \
  -o ${OUTPUT_FILE} \
  "find ${CORPORA_DIR} -type f -name "*.rel.xml" > ${OUTPUT_FILE}"

popd
