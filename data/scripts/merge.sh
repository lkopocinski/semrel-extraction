#!/bin/bash -eux

pushd "$(git rev-parse --show-toplevel)"

DATA_IN=$1
OUTPUT_PATH=$2

mkdir -p ${OUTPUT_PATH}


domain_out=81
mkdir -p ${OUTPUT_PATH}/${domain_out}
for domain in 82 83
do
    cat ${DATA_IN}/train/positive/${domain}.vectors >> ${OUTPUT_PATH}/${domain_out}/train.vectors
    cat ${DATA_IN}/valid/positive/${domain}.vectors >> ${OUTPUT_PATH}/${domain_out}/valid.vectors
    cat ${DATA_IN}/test/positive/${domain}.vectors >> ${OUTPUT_PATH}/${domain_out}/valid.vectors

    cat ${DATA_IN}/train/negative/${domain}.vectors >> ${OUTPUT_PATH}/${domain_out}/train.vectors
    cat ${DATA_IN}/valid/negative/${domain}.vectors >> ${OUTPUT_PATH}/${domain_out}/valid.vectors
    cat ${DATA_IN}/test/negative/${domain}.vectors >> ${OUTPUT_PATH}/${domain_out}/valid.vectors
done

for type in "train" "valid" "test"
do
    cat ${DATA_IN}/${type}/positive/${domain_out}.vectors >> ${OUTPUT_PATH}/${domain_out}/test.vectors
    cat ${DATA_IN}/${type}/negative/${domain_out}.vectors >> ${OUTPUT_PATH}/${domain_out}/test.vectors
done

shuf -o ${OUTPUT_PATH}/${domain_out}/train.vectors ${OUTPUT_PATH}/${domain_out}/train.vectors
shuf -o ${OUTPUT_PATH}/${domain_out}/valid.vectors ${OUTPUT_PATH}/${domain_out}/valid.vectors
shuf -o ${OUTPUT_PATH}/${domain_out}/valid.vectors ${OUTPUT_PATH}/${domain_out}/test.vectors

domain_out=82
mkdir -p ${OUTPUT_PATH}/${domain_out}
for domain in 81 83
do
    cat ${DATA_IN}/train/positive/${domain}.vectors >> ${OUTPUT_PATH}/${domain_out}/train.vectors
    cat ${DATA_IN}/valid/positive/${domain}.vectors >> ${OUTPUT_PATH}/${domain_out}/valid.vectors
    cat ${DATA_IN}/test/positive/${domain}.vectors >> ${OUTPUT_PATH}/${domain_out}/valid.vectors

    cat ${DATA_IN}/train/negative/${domain}.vectors >> ${OUTPUT_PATH}/${domain_out}/train.vectors
    cat ${DATA_IN}/valid/negative/${domain}.vectors >> ${OUTPUT_PATH}/${domain_out}/valid.vectors
    cat ${DATA_IN}/test/negative/${domain}.vectors >> ${OUTPUT_PATH}/${domain_out}/valid.vectors
done

for type in "train" "valid" "test"
do
    cat ${DATA_IN}/${type}/positive/${domain_out}.vectors >> ${OUTPUT_PATH}/${domain_out}/test.vectors
    cat ${DATA_IN}/${type}/negative/${domain_out}.vectors >> ${OUTPUT_PATH}/${domain_out}/test.vectors
done

shuf -o ${OUTPUT_PATH}/${domain_out}/train.vectors ${OUTPUT_PATH}/${domain_out}/train.vectors
shuf -o ${OUTPUT_PATH}/${domain_out}/valid.vectors ${OUTPUT_PATH}/${domain_out}/valid.vectors
shuf -o ${OUTPUT_PATH}/${domain_out}/valid.vectors ${OUTPUT_PATH}/${domain_out}/test.vectors

domain_out=83
mkdir -p ${OUTPUT_PATH}/${domain_out}
for domain in 81 82
do
    cat ${DATA_IN}/train/positive/${domain}.vectors >> ${OUTPUT_PATH}/${domain_out}/train.vectors
    cat ${DATA_IN}/valid/positive/${domain}.vectors >> ${OUTPUT_PATH}/${domain_out}/valid.vectors
    cat ${DATA_IN}/test/positive/${domain}.vectors >> ${OUTPUT_PATH}/${domain_out}/valid.vectors

    cat ${DATA_IN}/train/negative/${domain}.vectors >> ${OUTPUT_PATH}/${domain_out}/train.vectors
    cat ${DATA_IN}/valid/negative/${domain}.vectors >> ${OUTPUT_PATH}/${domain_out}/valid.vectors
    cat ${DATA_IN}/test/negative/${domain}.vectors >> ${OUTPUT_PATH}/${domain_out}/valid.vectors
done

for type in "train" "valid" "test"
do
    cat ${DATA_IN}/${type}/positive/${domain_out}.vectors >> ${OUTPUT_PATH}/${domain_out}/test.vectors
    cat ${DATA_IN}/${type}/negative/${domain_out}.vectors >> ${OUTPUT_PATH}/${domain_out}/test.vectors
done

shuf -o ${OUTPUT_PATH}/${domain_out}/train.vectors ${OUTPUT_PATH}/${domain_out}/train.vectors
shuf -o ${OUTPUT_PATH}/${domain_out}/valid.vectors ${OUTPUT_PATH}/${domain_out}/valid.vectors
shuf -o ${OUTPUT_PATH}/${domain_out}/valid.vectors ${OUTPUT_PATH}/${domain_out}/test.vectors

# Generate for default
mkdir -p ${OUTPUT_PATH}/all
for type in "train" "valid" "test"
do
    OUTPUT_FILE=${OUTPUT_PATH}/all/${type}.vectors
    cat ${DATA_IN}/${type}/*/* > ${OUTPUT_FILE}
    shuf -o ${OUTPUT_FILE} ${OUTPUT_FILE}
done

popd
