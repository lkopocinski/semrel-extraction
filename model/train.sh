#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

SCRIPTS_DIR="./model/scripts"
CONFIG="./model/config.yaml"

CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/train.py --config ${CONFIG}

popd
