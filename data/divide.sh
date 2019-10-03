#!/usr/bin/env bash

# Params
RES_DIR=generated
TRAIN_DIR=${RES_DIR}/train
VALID_DIR=${RES_DIR}/valid
TEST_DIR=${RES_DIR}/test
OUTPUT_DIR=sampled

# Initialization
declare -a types=("train" "valid" "test")
for dataset_type in "${types[@]}"
do
    # Declaration
    OUT_DIR=${OUTPUT_DIR}/${dataset_type}
    POSITIVE_DIR=${OUT_DIR}/positive
    NEGATIVE_DIR=${OUT_DIR}/negative
    SUBSTITUTED_DIR=${OUT_DIR}/negative/substituted

    mkdir -p ${MULTIWORD_DIR}
    mkdir -p ${SUBSTITUTED_DIR}
done

# Generation
python3.6 scripts/divide_datasets.py --positive_size 1500 --negative_size 3000 --source_dir ${TRAIN_DIR} --output_dir ${OUTPUT_DIR}/train
python3.6 scripts/divide_datasets.py --positive_size 500 --negative_size 1000 --source_dir ${VALID_DIR} --output_dir ${OUTPUT_DIR}/valid
python3.6 scripts/divide_datasets.py --positive_size 500 --negative_size 1000 --source_dir ${TEST_DIR} --output_dir ${OUTPUT_DIR}/test