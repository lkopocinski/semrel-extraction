
for file_nr in 52 54 55 81 82 83
do
	python3.6 gen_positive.py -d /home/lukaszkopocinski/Lukasz/SentiOne/korpusyneroweiaspektowe/inforex_export_${file_nr}/documents/ -c '['BRAND_NAME', 'PRODUCT_NAME']' > ${file_nr}.context
done

