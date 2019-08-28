DATA_DIR='data_model_6'

# Positive data
sort -u positive.vectors > positive.uniq.vectors
shuf positive.uniq.vectors > positive.uniq.shuf.vectors

file=positive.uniq.shuf.vectors
num_files=5

total_lines=$(wc -l <${file})
((lines_per_file = (total_lines + num_files - 1) / num_files))

split --lines=${lines_per_file} ${file}

cat xaa >> ${DATA_DIR}/train.vectors
cat xab >> ${DATA_DIR}/train.vectors
cat xac >> ${DATA_DIR}/train.vectors
cat xad >> ${DATA_DIR}/valid.vectors
cat xae >> ${DATA_DIR}/test.vectors


# Negative data
sort -u negative.vectors > negative.uniq.vectors
shuf negative.uniq.vectors > negative.uniq.shuf.vectors

file=negative.uniq.shuf.vectors
num_files=5

total_lines=$(wc -l <${file})
((lines_per_file = (total_lines + num_files - 1) / num_files))

split --lines=${lines_per_file} ${file}

cat xaa >> ${DATA_DIR}/train.vectors
cat xab >> ${DATA_DIR}/train.vectors
cat xac >> ${DATA_DIR}/train.vectors
cat xad >> ${DATA_DIR}/valid.vectors
cat xae >> ${DATA_DIR}/test.vectors


# Negative hard data
sort -u negative_hard.vectors > negative_hard.uniq.vectors
shuf negative_hard.uniq.vectors > negative_hard.uniq.shuf.vectors

head -139000 negative_hard.uniq.shuf.vectors >> ${DATA_DIR}/train.vectors
tail -1000 negative_hard.uniq.shuf.vectors >> ${DATA_DIR}/test_negative_hard.vectors


# Negative substituted data brand
shuf negative_substituted_brand.uniq.vectors > negative_substituted_brand.uniq.shuf.vectors

head -20000 negative_substituted_brand.uniq.shuf.vectors >> ${DATA_DIR}/train.vectors
tail -10000 negative_substituted_brand.uniq.shuf.vectors >> ${DATA_DIR}/test_negative_subst_brand.vectors

# Negative substituted data product
shuf negative_substituted_product.uniq.vectors > negative_substituted_product.uniq.shuf.vectors

head -20000 negative_substituted_product.uniq.shuf.vectors >> ${DATA_DIR}/train.vectors
tail -10000 negative_substituted_product.uniq.shuf.vectors >> ${DATA_DIR}/test_negative_subst_product.vectors


# Shuf data
shuf -o ${DATA_DIR}/train.vectors ${DATA_DIR}/train.vectors 
shuf -o ${DATA_DIR}/valid.vectors ${DATA_DIR}/valid.vectors 
shuf -o ${DATA_DIR}/test.vectors ${DATA_DIR}/test.vectors


# Remove temp data
rm positive.uniq.vectors
rm positive.uniq.shuf.vectors
rm negative.uniq.vectors
rm negative.uniq.shuf.vectors
rm negative_hard.uniq.vectors
rm negative_hard.uniq.shuf.vectors
rm negative_substituted_brand.uniq.shuf.vectors
rm negative_substituted_product.uniq.shuf.vectors
