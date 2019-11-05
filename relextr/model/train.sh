#!/usr/bin/env bash

# Params
DATASET_DIR=dataset
SCRIPTS_DIR=scripts
EPOCHS_QUANTITY=30
BATCH_SIZE=10
SAVE_MODEL_NAME='relextr_model.pt'
SENT_2_VEC_MODEL=$(pwd)/../../data/sent2vec/kgr10.bin
FASTTEXT_VEC_MODEL=$(pwd)/../../data/fasttext/

# Execution
CUDA_VISIBLE_DEVICES=5,6 python3.6 ${SCRIPTS_DIR}/train.py --epochs ${EPOCHS_QUANTITY} --batch_size ${BATCH_SIZE} --dataset_dir ${DATASET_DIR} --model_name ${SAVE_MODEL_NAME} --sent2vec ${SENT_2_VEC_MODEL}
