#!/usr/bin/env bash

# Params
RES_DIR=generated
TRAIN_DIR=${RES_DIR}/train
VALID_DIR=${RES_DIR}/valid
TEST_DIR=${RES_DIR}/test
OUTPUT_DIR=sampled

POSITIVE_SIZE=100
NEGATIVE_SIZE=100

mkdir -p ${OUTPUT_DIR}/positive
mkdir -p ${OUTPUT_DIR}/negative
mkdir -p ${OUTPUT_DIR}/negative/substituted


for path in ${TRAIN_DIR}
do
    python3.6 scripts/divide_datasets.py --positive_size ${POSITIVE_SIZE} --negative_size ${NEGATIVE_SIZE} --source_dir ${path} --output_dir ${OUTPUT_DIR}
done
