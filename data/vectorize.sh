#!/usr/bin/env bash

# Params
SCRIPTS_DIR=scripts
OPTIONS_FILE='/data2/piotrmilkowski/bilm-tf-data/e2000000/options.json'
WEIGHTS_FILE='/data2/piotrmilkowski/bilm-tf-data/e2000000/weights.hdf5'

SOURCE_PATH=''
RELATION_TYPE='in_relation'
#RELATION_TYPE='no_relation'

python3.6 ${SCRIPTS_DIR}/create_vectors.py -s ${SOURCE_PATH}.context -r ${RELATION_TYPE} -p ${OPTIONS_FILE} -w ${WEIGHTS_FILE}