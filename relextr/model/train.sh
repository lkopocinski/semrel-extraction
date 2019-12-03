#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

SCRIPTS_DIR="./relextr/model/scripts"
CONFIG="./relextr/model/config.yaml"

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 ${SCRIPTS_DIR}/train.py --config ${CONFIG}

popd
