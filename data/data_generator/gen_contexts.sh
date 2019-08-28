#!/bin/bash

file_nrs=(52 54 55 81 82 83)
prefix='negative'


for file_nr in ${file_nrs[*]}
do
	python3.6 gen_negative.py -d /home/lukaszkopocinski/Lukasz/SentiOne/korpusyneroweiaspektowe/inforex_export_${file_nr}/documents/ -c '['BRAND_NAME', 'PRODUCT_NAME']' > ${file_nr}.context

	echo $file_nr
done

for file_nr in ${file_nrs[*]}
do
	cat ${file_nr}.context >> ${prefix}.context
       echo $file_nr	
done

sort -u -o ${prefix}.context ${prefix}.context 
