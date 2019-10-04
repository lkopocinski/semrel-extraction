#!/usr/bin/env bash

# Params
SOURCE_DIR=generated
TARGET_DIR=sampled
SCRIPTS_DIR=scripts

# Initialization
declare -a types=("train" "valid" "test")
for type in "${types[@]}"
do
    # Declaration
    OUT_DIR=${TARGET_DIR}/${type}
    POSITIVE_DIR=${OUT_DIR}/positive
    NEGATIVE_DIR=${OUT_DIR}/negative
    SUBSTITUTED_DIR=${OUT_DIR}/negative/substituted

    mkdir -p ${POSITIVE_DIR} ${SUBSTITUTED_DIR}
done

# Sample
python3.6 ${SCRIPTS_DIR}/sample_datasets.py --positive_size 1500 --negative_size 3000 --source_dir ${SOURCE_DIR}/train --output_dir ${TARGET_DIR}/train
python3.6 ${SCRIPTS_DIR}/sample_datasets.py --positive_size 500 --negative_size 1000 --source_dir ${SOURCE_DIR}/valid --output_dir ${TARGET_DIR}/valid
python3.6 ${SCRIPTS_DIR}/sample_datasets.py --positive_size 500 --negative_size 1000 --source_dir ${SOURCE_DIR}/test --output_dir ${TARGET_DIR}/test
