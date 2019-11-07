#!/bin/bash -eux

# Script takes previously split data and generate examples used to train classifier.
# Returns files in context style data format.

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./data/splits"
OUTPUT_PATH="./data/generated"
SCRIPTS_DIR="./data/scripts"

mkdir -p ${OUTPUT_PATH}

dvc run \
-d ${DATA_IN} \
-d ${SCRIPTS_DIR}/generate_positive_from_corpora.py \
-o ${OUTPUT_PATH} \
${SCRIPTS_DIR}/generate_positive_from_corpora.py --data-in ${DATA_IN} \
                                                 --output-path ${OUTPUT_PATH}

popd

#OUT_DIR=${OUTPUT_PATH}/${type}
#NEGATIVE_DIR=${OUT_DIR}/negative
#
#python3.6 ${SCRIPTS_DIR}/generate_negative_from_corpora.py --list_file ${list} --channels "${CHANNELS}" > ${NEGATIVE_DIR}/${name}.context
#sort -u -o ${NEGATIVE_DIR}/${name}.context ${NEGATIVE_DIR}/${name}.context
