#!/usr/bin/env bash

DATA_IN=$1
OUTPUT_PATH=$2

declare -a set_types=("train" "valid" "test")
for type in "${set_types[@]}"
do
    OUTPUT_FILE=${OUTPUT_PATH}/${type}.vectors
    cat ${DATA_IN}/${type}/*/* > ${OUTPUT_FILE}
    shuf -o ${OUTPUT_FILE} ${OUTPUT_FILE}
done
