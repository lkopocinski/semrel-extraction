#!/bin/bash -eux

# Script runs test training session

pushd "$(git rev-parse --show-toplevel)"

DATA_IN="./relextr/model/dataset"
SCRIPTS_DIR="./relextr/model/scripts"

SENT2VEC_MODEL="./data/sent2vec/kgr10.bin"
FASTTEXT_MODEL="./data/fasttext/kgr10.plain.lemma.skipgram.dim300.neg10.bin"


dvc run \
-d ${DATA_IN} \
-d ${SCRIPTS_DIR}/test.py \
-d ${SCRIPTS_DIR}/relnet.py \
-d ${SCRIPTS_DIR}/utils/engines.py \
-d ${SCRIPTS_DIR}/utils/utils.py \
-d ${SCRIPTS_DIR}/utils/metrics.py \
-d ${SCRIPTS_DIR}/utils/batches.py \
-M metrics_test.txt \
-f test.dvc \
CUDA_VISIBLE_DEVICES=7,8,9,10 ${SCRIPTS_DIR}/test.py --data-in ${DATA_IN} \
                                                     --model-name 'relextr_model.pt' \
                                                     --batch-size 20 \
                                                     --tracking-uri 'http://10.17.50.132:8080' \
                                                     --experiment-name 'plain'
#                                                     --vectorizer 'fasttext' \
#                                                     --vectors-model ${FASTTEXT_MODEL} \

popd
