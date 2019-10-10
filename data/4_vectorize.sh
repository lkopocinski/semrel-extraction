#!/usr/bin/env bash

# Params
SOURCE_DIR=sampled
TARGET_DIR=vectors
SCRIPTS_DIR=scripts

declare -a types=("train" "valid" "test")
for type in "${types[@]}"
do
    TARGET_TMP=${TARGET_DIR}/tmp/${type}
    mkdir -p ${TARGET_TMP}
    cat ${SOURCE_DIR}/${type}/positive/*.sampled > ${TARGET_TMP}/positive.context
    cat ${SOURCE_DIR}/${type}/negative/*.sampled > ${TARGET_TMP}/negative.context
    cat ${SOURCE_DIR}/${type}/negative/substituted/*.sampled >> ${TARGET_TMP}/negative.context
done

OPTIONS_FILE='/data2/piotrmilkowski/bilm-tf-data/e2000000/options.json'
WEIGHTS_FILE='/data2/piotrmilkowski/bilm-tf-data/e2000000/weights.hdf5'

for type in "${types[@]}"
do
    SOURCE_TMP=${TARGET_DIR}/tmp/${type}
    FILE_POSITIVE=${SOURCE_TMP}/positive.context
    FILE_NEGATIVE=${SOURCE_TMP}/negative.context

    mkdir -p ${TARGET_DIR}/${type}
    python3.6 ${SCRIPTS_DIR}/create_vectors.py --source_file ${FILE_POSITIVE} --relation_type "in_relation" --options ${OPTIONS_FILE} --weights ${WEIGHTS_FILE} > ${TARGET_DIR}/${type}/positive.vectors
    python3.6 ${SCRIPTS_DIR}/create_vectors.py --source_file ${FILE_NEGATIVE} --relation_type "no_relation" --options ${OPTIONS_FILE} --weights ${WEIGHTS_FILE} > ${TARGET_DIR}/${type}/negative.vectors
done
