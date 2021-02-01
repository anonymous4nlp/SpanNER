#use_spanLen=$0
#use_morphology=$1
#use_span_weight=$2
#neg_span_weight=$3
#gpus=$4

bash conll03_spanPred.sh  False False False False 1 "0,"
bash conll03_spanPred.sh  True False False False 1 "1,"
bash conll03_spanPred.sh  False True False False 1 "2,"
bash conll03_spanPred.sh  True True False False 1 "3,"
bash conll03_spanPred.sh  False False True False 1 "4,"
bash conll03_spanPred.sh  False True True False 1 "5,"
bash conll03_spanPred.sh  True True True False 1 "6,"

