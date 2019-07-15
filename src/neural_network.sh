#!/usr/bin/env bash

# load configuration
source configuration.sh
export NUM_INDEPENDENT=$(python -c "execd = {} ; exec(open(\""$CONFIGURATION_FILE"\").read(), execd) ; print execd[\"NUM_INDEPENDENT\"]")

# clean
rm -f "neural_network_"$ABM"_cross_validation.txt"

# evaluate
export K=10
export TESTS_PER_FOLD=10

for i in $(seq 0 $(expr "$K" - 1))
do
    export SPLIT_NUMBER=$i

    # train network and suggest slps
    python3 neural_network.py --abm $ABM --numIndependent $NUM_INDEPENDENT --numDependent $NUM_DEPENDENT --fmOnly 0 --rmOnly 0 --splitNumber $SPLIT_NUMBER --testCase 1

    # simulate behavior
    ./simulate.sh

    # compute distances
    python get_neural_network_error.py
done
