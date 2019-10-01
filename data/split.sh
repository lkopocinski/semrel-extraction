#!/usr/bin/env bash

# Params
ROOT_PATH=./korpusy
FILES_NR=(81 82 83)
SCRIPTS_DIR=scripts
SAVE_DIR=splits

# Initialization
mkdir -p ${SAVE_DIR}
mkdir -p ${SAVE_DIR}/train
mkdir -p ${SAVE_DIR}/valid
mkdir -p ${SAVE_DIR}/test

# Split
for nr in ${FILES_NR[*]}
do
    FILES_PATH=${ROOT_PATH}/inforex_export_${nr}/documents/

    python3.6 ${SCRIPTS_DIR}/split_dataset_files.py -d ${FILES_PATH} -o ${SAVE_DIR} -p ${nr}
done
