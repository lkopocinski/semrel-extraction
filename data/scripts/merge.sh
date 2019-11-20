#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

DATA_IN=$1
OUTPUT_PATH=$2
MODE=$3

mkdir -p ${OUTPUT_PATH}

if [[ ${MODE} == "one_out" ]]; then

    domain_out=82
    for domain in 81 83
    do
        cat ${DATA_IN}/train/positive/${domain}.vectors >> ${OUTPUT_PATH}/train.vectors
        cat ${DATA_IN}/valid/positive/${domain}.vectors >> ${OUTPUT_PATH}/valid.vectors
        cat ${DATA_IN}/test/positive/${domain}.vectors >> ${OUTPUT_PATH}/valid.vectors

        cat ${DATA_IN}/train/negative/${domain}.vectors >> ${OUTPUT_PATH}/train.vectors
        cat ${DATA_IN}/valid/negative/${domain}.vectors >> ${OUTPUT_PATH}/valid.vectors
        cat ${DATA_IN}/test/negative/${domain}.vectors >> ${OUTPUT_PATH}/valid.vectors		
    done
    
    for type in "train" "valid" "test"
    do
        cat ${DATA_IN}/${type}/positive/${domain_out}.vectors >> ${OUTPUT_PATH}/test.vectors
        cat ${DATA_IN}/${type}/negative/${domain_out}.vectors >> ${OUTPUT_PATH}/test.vectors
    done

    shuf -o ${OUTPUT_PATH}/train.vectors ${OUTPUT_PATH}/train.vectors
    shuf -o ${OUTPUT_PATH}/valid.vectors ${OUTPUT_PATH}/valid.vectors
    shuf -o ${OUTPUT_PATH}/valid.vectors ${OUTPUT_PATH}/test.vectors

else

    for type in "train" "valid" "test"
    do
        OUTPUT_FILE=${OUTPUT_PATH}/${type}.vectors
        cat ${DATA_IN}/${type}/*/* > ${OUTPUT_FILE}
        shuf -o ${OUTPUT_FILE} ${OUTPUT_FILE}
    done

fi

popd
