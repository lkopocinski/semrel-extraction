#!/usr/bin/env bash

# Params
SCRIPTS_DIR=scripts
SOURCE_DIR=sampled
TARGET_DIR=vectors

declare -a types=("train" "valid" "test")
for type in "${types[@]}"
do
    TARGET_TMP=${TARGET_DIR}/tmp/${type}
    mkdir -p ${TARGET_TMP}
    cat ${SOURCE_DIR}/${type}/positive/*.sampled > ${TARGET_TMP}/positive.context
    cat ${SOURCE_DIR}/${type}/negative/*.sampled > ${TARGET_TMP}/negative.context
    cat ${SOURCE_DIR}/${type}/negative/substituted/*.sampled > ${TARGET_TMP}/negative.context
done

OPTIONS_FILE='/data2/piotrmilkowski/bilm-tf-data/e2000000/options.json'
WEIGHTS_FILE='/data2/piotrmilkowski/bilm-tf-data/e2000000/weights.hdf5'

for type in "${types[@]}"
do
    SRC_TMP=${TARGET_DIR}/tmp/${type}
    FILE_POSITIVE=${SRC_TMP}/positive.context
    FILE_NEGATIVE=${SRC_TMP}/negative.context

    mkdir -p ${TARGET_DIR}/${type}
    python3.6 ${SCRIPTS_DIR}/create_vectors.py -s ${FILE_POSITIVE} -r "in_relation" -p ${OPTIONS_FILE} -w ${WEIGHTS_FILE} > ${TARGET_DIR}/${type}/positive.vectors
    python3.6 ${SCRIPTS_DIR}/create_vectors.py -s ${FILE_NEGATIVE} -r "no_relation" -p ${OPTIONS_FILE} -w ${WEIGHTS_FILE} > ${TARGET_DIR}/${type}/negative.vectors
done
