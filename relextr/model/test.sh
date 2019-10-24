#!/usr/bin/env bash

# Params
DATASET_DIR=dataset
SCRIPTS_DIR=scripts
BATCH_SIZE=20
MODEL_NAME='relextr_model.pt'
SENT_2_VEC_MODEL=$(pwd)/../../data/sent2vec/kgr10.bin

# Execution
CUDA_VISIBLE_DEVICES=5,6 python3.6 ${SCRIPTS_DIR}/test.py --batch_size ${BATCH_SIZE} --dataset_dir ${DATASET_DIR} --model_name ${MODEL_NAME} --sent2vec ${SENT_2_VEC_MODEL}
