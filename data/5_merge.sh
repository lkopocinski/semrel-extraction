#!/bin/bash -eux

# Script concatenates vector files into one shuffled file

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./data/vectors"
OUTPUT_PATH="./relextr/model/dataset"
SCRIPTS_DIR="./data/scripts"

mkdir -p ${OUTPUT_PATH}

dvc run \
-d ${DATA_IN} \
-d ${SCRIPTS_DIR}/merge.sh \
-o ${OUTPUT_PATH} \
${SCRIPTS_DIR}/merge.sh ${DATA_IN} ${OUTPUT_PATH} "default" 

popd
