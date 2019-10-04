#!/usr/bin/env bash

# Script takes previously split data and generate examples used to train classifier.
# Returns files in context like data format.

# Params
SOURCE_DIR=splits
TARGET_DIR=generated
SCRIPTS_DIR=scripts
CHANNELS='['BRAND_NAME', 'PRODUCT_NAME']'
BRANDS_SAMPLE=5

# Generate
declare -a types=("train" "valid" "test")
for type in "${types[@]}"
do
    # Declaration
    OUT_DIR=${TARGET_DIR}/${type}
    POSITIVE_DIR=${OUT_DIR}/positive
    NEGATIVE_DIR=${OUT_DIR}/negative
    MULTIWORD_DIR=${OUT_DIR}/positive/multiword
    SUBSTITUTED_DIR=${OUT_DIR}/negative/substituted

    mkdir -p ${MULTIWORD_DIR} ${SUBSTITUTED_DIR}

    for list in ${SOURCE_DIR}/${type}/*.list
    do
        echo ${list}
        name=$(echo ${list} | cut -d"/" -f3 | cut -d"." -f1)
	
	    python3.6 ${SCRIPTS_DIR}/generate_positive_from_corpora.py --list_file ${list} --channels "${CHANNELS}" > ${POSITIVE_DIR}/${name}.context
        sort -u -o ${POSITIVE_DIR}/${name}.context ${POSITIVE_DIR}/${name}.context

        python3.6 ${SCRIPTS_DIR}/generate_negative_from_corpora.py --list_file ${list} --channels "${CHANNELS}" > ${NEGATIVE_DIR}/${name}.context
        sort -u -o ${NEGATIVE_DIR}/${name}.context ${NEGATIVE_DIR}/${name}.context

        # Intermediate step
        python3.6 ${SCRIPTS_DIR}/generate_positive_from_corpora.py --list_file ${list} --channels "${CHANNELS}" --multiword True > ${MULTIWORD_DIR}/${name}.context
        sort -u -o ${MULTIWORD_DIR}/${name}.context ${MULTIWORD_DIR}/${name}.context

        python3.6 ${SCRIPTS_DIR}/generate_negative_substitition.py --source_file ${MULTIWORD_DIR}/${name}.context --sample_size ${BRANDS_SAMPLE} > ${SUBSTITUTED_DIR}/${name}.context
        sort -u -o ${SUBSTITUTED_DIR}/${name}.context  ${SUBSTITUTED_DIR}/${name}.context
    done
done
