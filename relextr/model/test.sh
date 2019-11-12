#!/bin/bash -eux

# Script runs test training session

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./relextr/model/dataset"
SCRIPTS_DIR="./relextr/model/scripts"

SENT2VEC_MODEL="./data/sent2vec/kgr10.bin"
FASTTEXT_MODEL="./data/fasttext/kgr10.plain.lemma.skipgram.dim300.neg10.bin"


dvc run \
-d -d ${DATA_IN} \
-d ${SCRIPTS_DIR}/test.py \
CUDA_VISIBLE_DEVICES=5,6 ${SCRIPTS_DIR}/test.py --data-in ${DATA_IN} \
                                                --

--batch_size 20 --dataset_dir ${DATA_IN} --model_name ${MODEL_NAME} --sent2vec ${SENT_2_VEC_MODEL}
