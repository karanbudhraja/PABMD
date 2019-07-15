#!/usr/bin/env bash
# load configuration
source configuration.sh

# suffix based on random baseline or not
export SUFFIX="_cross_validation"
if [ $RANDOM_BASELINE == 1 ]; then
    export SUFFIX=$SUFFIX"_random_baseline"
fi
export SUFFIX=$SUFFIX".txt"

# remove old results
rm -f evaluation_distance_$ABM$SUFFIX

# run for all test entries in each fold
# PREDICTION_NAME is dependent on the demonstration folder name
# not that the actual demonstration in that folder is not used
# we overwrite "queried_slps.txt"
export K=1
export PREDICTION_NAME=1
export TESTS_PER_FOLD=2000

for i in $(seq 0 $(expr "$K" - 1))
do
    export SPLIT_FILE_NAME_PREFIX=$ABM

    if [ $IMAGES == 1 ]; then
	export LOWER_CASE=$(echo $DESCRIPTOR_SETTING | tr '[:upper:]' '[:lower:]')
	export SPLIT_FILE_NAME_PREFIX=$SPLIT_FILE_NAME_PREFIX"_"$LOWER_CASE
    fi
    
    export SPLIT_TRAIN_FILE_NAME=$SPLIT_FILE_NAME_PREFIX"_split_"$i"_train.txt"
    export SPLIT_TEST_FILE_NAME=$SPLIT_FILE_NAME_PREFIX"_split_"$i"_test.txt"

    head -n $TESTS_PER_FOLD "../data/domaindata/cross_validation/"$SPLIT_TEST_FILE_NAME | while read line; do
	# get demonstration from line
	python get_test_demonstration.py $line

	# run experiment
	# simulate results
	# get prediction
	./lfd.sh && ./simulate.sh && ./filter.sh

	# evaluate distance between prediction and queried slps
	export DISTANCE=$(python get_output_difference.py)
	echo $DISTANCE >> evaluation_distance_$ABM$SUFFIX
    done
done
