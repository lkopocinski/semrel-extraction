#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

SCRIPTS_DIR="./semrel/model/scripts"
CONFIG="./semrel/model/config.yaml"

CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/train.py --config ${CONFIG}

popd
