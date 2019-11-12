#!/bin/bash -eux

# TODO: description

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./relextr/model/dataset"
SCRIPTS_DIR="./relextr/model/scripts"

EPOCHS_QUANTITY=30
BATCH_SIZE=10

SENT2VEC_MODEL="./data/sent2vec/kgr10.bin"
FASTTEXT_MODEL="./data/fasttext/kgr10.plain.lemma.skipgram.dim300.neg10.bin"

dvc run \
-d ${DATA_IN} \
-d ${SCRIPTS_DIR}/train.py \
-d ${SCRIPTS_DIR}/relnet.py \
-d ${SCRIPTS_DIR}/engines.py \
-d ${SCRIPTS_DIR}/utils.py \
-d ${SCRIPTS_DIR}/metrics.py \
-M metrics.txt \
-f train.dvc \
CUDA_VISIBLE_DEVICES=5,6 ${SCRIPTS_DIR}/train.py --data-in ${DATA_IN} \
                                                 --save-model-name 'relextr_model.pt' \
                                                 --batch-size 10 \
                                                 --epochs 30 \
                                                 --vectorizer 'default' \
                                                 --vectors-model '' \
                                                 --tracking-uri 'http://0.0.0.0:5001' \
                                                 --experiment-name 'no_experiment'

popd