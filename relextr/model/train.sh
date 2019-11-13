#!/bin/bash -eux

# Script runs model training session

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./relextr/model/dataset"
SCRIPTS_DIR="./relextr/model/scripts"

SENT2VEC_MODEL="./data/sent2vec/kgr10.bin"
FASTTEXT_MODEL="./data/fasttext/kgr10.plain.lemma.skipgram.dim300.neg10.bin"

dvc run \
-d ${DATA_IN} \
-d ${SCRIPTS_DIR}/train.py \
-d ${SCRIPTS_DIR}/relnet.py \
-d ${SCRIPTS_DIR}/utils/engines.py \
-d ${SCRIPTS_DIR}/utils/utils.py \
-d ${SCRIPTS_DIR}/utils/metrics.py \
-d ${SCRIPTS_DIR}/utils/batches.py \
-M metrics.txt \
-f train.dvc \
CUDA_VISIBLE_DEVICES=7,8,9,10 ${SCRIPTS_DIR}/train.py --data-in ${DATA_IN} \
                                                 --save-model-name 'relextr_model.pt' \
                                                 --batch-size 10 \
                                                 --epochs 10 \
                                                 --tracking-uri 'http://10.17.50.132:8080' \
                                                 --experiment-name 'no_experiment' \
                                                 --vectorizer 'sent2vec' \
                                                 --vectors-model ${SENT2VEC_MODEL} \

popd
