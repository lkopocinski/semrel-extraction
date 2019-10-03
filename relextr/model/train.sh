#!/usr/bin/env bash

# Params
EPOCHS_QUANTITY=30
BATCH_SIZE=10
SAVE_MODEL_NAME='model_temp.pt'

DATASET_DIR='dataset'
VECTORS_DIR=../../data/vectors

# Build dataset
declare -a types=("train" "valid" "test")
for type in "${types[@]}"
do
    cat ${VECTORS_DIR}/${type}/negative.vectors > ${DATASET_DIR}/${type}.vectors
    cat ${VECTORS_DIR}/${type}/positive.vectors >> ${DATASET_DIR}/${type}.vectors
    shuf -o ${DATASET_DIR}/${type}.vectors ${DATASET_DIR}/${type}.vectors
done

# Execution
python3.6 train.py --epochs ${EPOCHS_QUANTITY} --batch_size ${BATCH_SIZE} --datasets_dir ${DATASET_DIR} --model_name ${SAVE_MODEL_NAME}