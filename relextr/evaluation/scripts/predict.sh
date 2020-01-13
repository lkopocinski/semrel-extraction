#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

FILEINDEX_PATH=$1
MODEL_PATH="./all.pt"
SCRIPTS_DIR="./relextr/evaluation/scripts"

mkdir -p ${OUTPUT_PATH}

CUDA_VISIBLE_DEVICES=0,1 ${SCRIPTS_DIR}/predict.py \
  --net_model ${MODEL_PATH} \
  --elmo_model "data/elmo/options.json" "data/elmo/weights.hdf5" \
  --fasttext_model "data/fasttext/kgr10.plain.skipgram.dim300.neg10.bin" \
  --fileindex ${FILEINDEX_PATH}

popd
