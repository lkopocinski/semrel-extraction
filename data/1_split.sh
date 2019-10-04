#!/usr/bin/env bash

# Script takes files from provided corpora and splits them into train, valid, test
# disjoint data sets list. It means each one of the splits contains different files.

# Params
SOURCE_DIR=korpusy
TARGET_DIR=splits
SCRIPTS_DIR=scripts
FILES_NR=(81 82 83)

# Initialization
mkdir -p ${TARGET_DIR}/train ${TARGET_DIR}/valid ${TARGET_DIR}/test

# Split
for nr in ${FILES_NR[*]}
do
    FILES_PATH=$(pwd)/${SOURCE_DIR}/inforex_export_${nr}/documents/

    python3.6 ${SCRIPTS_DIR}/split_dataset_files.py --source_dir ${FILES_PATH} --target_dir ${TARGET_DIR} --prefix ${nr}
done
