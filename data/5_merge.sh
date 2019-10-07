#!/usr/bin/env bash

# Params
SOURCE_DIR=vectors
TARGET_DIR=../relextr/model/dataset

mkdir -p ${TARGET_DIR}


# Merge
declare -a types=("train" "valid" "test")
for type in "${types[@]}"
do
    cat ${SOURCE_DIR}/${type}/negative.vectors > ${TARGET_DIR}/${type}.vectors
    cat ${SOURCE_DIR}/${type}/positive.vectors >> ${TARGET_DIR}/${type}.vectors
    shuf -o ${TARGET_DIR}/${type}.vectors ${TARGET_DIR}/${type}.vectors
done

shuf ${SOURCE_DIR}/train/negative.hard.vectors | head -2000 >> ${TARGET_DIR}/train.vectors
shuf -o ${TARGET_DIR}/train.vectors ${TARGET_DIR}/train.vectors


