#!/usr/bin/env bash

# Params
EPOCHS_QUANTITY=30
BATCH_SIZE=10
SAVE_MODEL_NAME='model_temp.pt'
DATASETS_DIR='datasets'

# Execution
python3.6 train.py --epochs $EPOCHS_QUANTITY} --batch_size ${BATCH_SIZE} --datasets_dir --model_name ${SAVE_MODEL_NAME}