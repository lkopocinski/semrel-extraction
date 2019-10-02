#!/usr/bin/env bash

# Params
RES_DIR=generated
TRAIN_DIR=${RES_DIR}/train
VALID_DIR=${RES_DIR}/valid
TEST_DIR=${RES_DIR}/test

POSITIVE_SIZE=100
NEGATIVE_SIZE=100


for path in TRAIN_DIR VALID_DIR TEST_DIR
do
    python3.6 divide_datsets.py --positive_size ${POSITIVE_SIZE} --negative_size ${NEGATIVE_SIZE} --path ${path}
done