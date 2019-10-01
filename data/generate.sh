#!/usr/bin/env bash

# Params
CHANNELS='['BRAND_NAME', 'PRODUCT_NAME']'
SCRIPTS_DIR=scripts
SPLITS_DIR=splits
RES_DIR=generated
BRANDS_SAMPLE=5

# Generate
declare -a types=("train" "valid" "test")
for dataset_type in "${types[@]}"
do
    # Declaration
    OUT_DIR=${RES_DIR}/${dataset_type}
    POSITIVE_DIR=${OUT_DIR}/positive
    NEGATIVE_DIR=${OUT_DIR}/negative
    MULTIWORD_DIR=${OUT_DIR}/positive/multiword
    SUBSTITUTED_DIR=${OUT_DIR}/negative/substituted

    mkdir -p ${MULTIWORD_DIR}
    mkdir -p ${SUBSTITUTED_DIR}

    DATASET_LIST_PATH=${SPLITS_DIR}/${dataset_type}
    for files_list in ${DATASET_LIST_PATH}/*.list
    do
	    echo ${files_list}
        prefix=$(echo ${files_list} | cut -d"/" -f3 | cut -d"." -f1)

	    python3.6 ${SCRIPTS_DIR}/generate_positive_from_corpora.py -d ${files_list} -c "${CHANNELS}" > ${POSITIVE_DIR}/${prefix}.context
        sort -u -o ${POSITIVE_DIR}/${nr}.context ${POSITIVE_DIR}/${nr}.context

        python3.6 ${SCRIPTS_DIR}/generate_negative_from_corpora.py -d ${files_list} -c "${CHANNELS}" > ${NEGATIVE_DIR}/${prefix}.context
        sort -u -o ${NEGATIVE_DIR}/${prefix}.context ${NEGATIVE_DIR}/${prefix}.context

        python3.6 ${SCRIPTS_DIR}/generate_positive_from_corpora.py -d ${files_list} -c "${CHANNELS}" --multiword True > ${MULTIWORD_DIR}/${prefix}.context
        sort -u -o ${MULTIWORD_DIR}/${prefix}.context ${MULTIWORD_DIR}/${prefix}.context

        python3.6 ${SCRIPTS_DIR}/generate_negative_substitition.py -s ${MULTIWORD_DIR}/${nr}.context --sample_size ${BRANDS_SAMPLE} > ${SUBSTITUTED_DIR}/${prefix}.context
        sort -u -o ${SUBSTITUTED_DIR}/${prefix}.context  ${SUBSTITUTED_DIR}/${prefix}.context
    done
done
