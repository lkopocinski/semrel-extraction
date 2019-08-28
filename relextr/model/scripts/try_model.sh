FILE=$1
DIR='predictions'

cat $FILE | python3.6 predict.py > $DIR/$FILE.predict
grep -P 'pred: in_relation\ttrue: no_relation' $DIR/$FILE.predict > $DIR/$FILE.grep

all_examples=$(wc -l $DIR/$FILE.predict)
false_predictions=$(wc -l $DIR/$FILE.grep)

echo $all_examples >> $DIR/$FILE.grep
echo $false_predictions >> $DIR/$FILE.grep
echo "Done"

