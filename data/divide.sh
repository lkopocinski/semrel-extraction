#!/usr/bin/env bash

# Params
RES_DIR=generated
TRAIN_DIR=${RES_DIR}/train
VALID_DIR=${RES_DIR}/valid
TEST_DIR=${RES_DIR}/test


for dataset_type in TRAIN_DIR VALID_DIR TEST_DIR
do
    python3.6 divide_datsets.py --positive_quantity --negative_quantity --path
done