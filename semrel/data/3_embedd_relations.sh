#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

SCRIPTS_DIR="./semrel/data/data/scripts/cli"
MAPS_DIR="./semrel/data/data/maps"

INPUT_PATH="./semrel/data/data/relations/relations.tsv"
OUTPUT_DIR="./semrel/data/data/vectors"

mkdir -p ${OUTPUT_DIR}

dvc run \
  -d ${INPUT_PATH} \
  -d ${SCRIPTS_DIR}/embedd_relations.py \
  -d ${MAPS_DIR} \
  -o ${OUTPUT_DIR}/elmo.rel.keys \
  -o ${OUTPUT_DIR}/elmo.rel.pt \
  -o ${OUTPUT_DIR}/fasttext.rel.keys \
  -o ${OUTPUT_DIR}/fasttext.rel.pt \
  -o ${OUTPUT_DIR}/retrofit.rel.keys \
  -o ${OUTPUT_DIR}/retrofit.rel.pt \
  -f vectors.dvc
  CUDA_VISIBLE_DEVICES=0 ${SCRIPTS_DIR}/embedd_relations.py --input-path ${INPUT_PATH} \
                                                            --elmo-map "${MAPS_DIR}/elmo.map.keys" "${MAPS_DIR}/elmo.map.pt" \
                                                            --fasttext-map "${MAPS_DIR}/fasttext.map.keys" "${MAPS_DIR}/fasttext.map.pt" \
                                                            --retrofit-map "${MAPS_DIR}/retrofit.map.keys" "${MAPS_DIR}/retrofit.map.pt" \
                                                            --output-dir ${OUTPUT_DIR}

popd
