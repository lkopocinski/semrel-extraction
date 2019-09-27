#!/usr/bin/env bash

# Params
ROOT_PATH=./korpusy
FILES_NR=(52 54 55 81 82 83)
CHANNELS='['BRAND_NAME', 'PRODUCT_NAME']'
SCRIPTS_DIR=scripts
RES_DIR=generated
BRANDS_SAMPLE=5

# Initialization
mkdir -p ${RES_DIR}/positive
mkdir -p ${RES_DIR}/negative
mkdir -p ${RES_DIR}/positive/multiword
mkdir -p ${RES_DIR}/negative/substituted

# Generate
for nr in ${FILES_NR[*]}
do
    FILES_PATH=${ROOT_PATH}/inforex_export_${nr}/documents/

    python3.6 ${SCRIPTS_DIR}/generate_positive_from_corpora.py -d ${FILES_PATH} -c "${CHANNELS}" > ${RES_DIR}/positive/${nr}.context
    sort -u ${RES_DIR}/positive/${nr}.context > ${RES_DIR}/positive/${nr}.context.u

    python3.6 ${SCRIPTS_DIR}/generate_negative_from_corpora.py -d ${FILES_PATH} -c "${CHANNELS}" > ${RES_DIR}/negative/${nr}.context
    sort -u ${RES_DIR}/negative/${nr}.context > ${RES_DIR}/negative/${nr}.context.u

    python3.6 ${SCRIPTS_DIR}/generate_positive_from_corpora.py -d ${FILES_PATH} -c "${CHANNELS}" --multiword True > ${RES_DIR}/positive/multiword/${nr}.context
    sort -u ${RES_DIR}/positive/multiword/${nr}.context > ${RES_DIR}/positive/multiword/${nr}.context.u

    python3.6 ${SCRIPTS_DIR}/generate_negative_substitition.py -s ${RES_DIR}/positive/multiword/${nr}.context --sample_size ${BRANDS_SAMPLE} > ${RES_DIR}/negative/substituted/${nr}.context
    sort -u ${RES_DIR}/negative/substituted/${nr}.context > ${RES_DIR}/negative/substituted/${nr}.context.u

done