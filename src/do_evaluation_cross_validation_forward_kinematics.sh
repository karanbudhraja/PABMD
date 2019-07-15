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
#export TESTS_PER_FOLD=1500
export TESTS_PER_FOLD=2000

for i in $(seq 0 $(expr "$K" - 1))
do
    export i="0000"
    echo $i
    
    export SPLIT_FILE_NAME_PREFIX=$ABM

    export SPLIT_TRAIN_FILE_NAME=$SPLIT_FILE_NAME_PREFIX"_train_set_"$i".txt"
    export SPLIT_TEST_FILE_NAME=$SPLIT_FILE_NAME_PREFIX"_test_set_"$i".txt"

    head -n $TESTS_PER_FOLD "../../_swarm-lfd-data/forward_kinematics/test/"$SPLIT_TEST_FILE_NAME | while read line; do
	# get demonstration from line
	python get_test_demonstration.py $line
	
	# run experiment
	./lfd.sh

	# scale suggested alps
	# compute difference from test data
	export DISTANCE=$(python compute_alps_scaled_difference.py $line)
	echo $DISTANCE >> evaluation_distance_$ABM$SUFFIX
    done
done
