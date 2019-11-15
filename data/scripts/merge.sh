#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

DATA_IN=$1
OUTPUT_PATH=$2

mkdir -p ${OUTPUT_PATH}

#declare -a set_types=("train" "valid" "test")
#for type in "${set_types[@]}"
#do
#    OUTPUT_FILE=${OUTPUT_PATH}/${type}.vectors
#    cat ${DATA_IN}/${type}/*/* > ${OUTPUT_FILE}
#    shuf -o ${OUTPUT_FILE} ${OUTPUT_FILE}
#done


domain_out=83
declare -a domains=("81" "82")
for domain in "${domains[@]}"
do
	cat ${DATA_IN}/train/positive/${domain}.vectors >> ${OUTPUT_PATH}/train.vectors
	cat ${DATA_IN}/valid/positive/${domain}.vectors >> ${OUTPUT_PATH}/valid.vectors
	cat ${DATA_IN}/test/positive/${domain}.vectors >> ${OUTPUT_PATH}/valid.vectors
	
	cat ${DATA_IN}/train/negative/${domain}.vectors >> ${OUTPUT_PATH}/train.vectors
	cat ${DATA_IN}/valid/negative/${domain}.vectors >> ${OUTPUT_PATH}/valid.vectors
	cat ${DATA_IN}/test/negative/${domain}.vectors >> ${OUTPUT_PATH}/valid.vectors
done

declare -a set_types=("train" "valid" "test")
for type in "${set_types[@]}"
do
	cat ${DATA_IN}/${type}/positive/${domain_out}.vectors >> ${OUTPUT_PATH}/test.vectors
	cat ${DATA_IN}/${type}/negative/${domain_out}.vectors >> ${OUTPUT_PATH}/test.vectors
done

popd
