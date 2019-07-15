#!/usr/bin/env bash
# load configuration
source configuration.sh

# run multiple times to generate multiple sets of images
export RUNS_PER_INPUT=10

# remove existing data
rm -f suggested_alps.txt
rm -f suggested_alps_fm_prediction_error.txt

for i in $(seq "$RUNS_PER_INPUT")
do
    # run experiment
    echo "=>running experiment"
    export ALP_CONFIGURATIONS=$(python lfd.py)
    echo $ALP_CONFIGURATIONS >> suggested_alps.txt
    
    # circumvent excessive prints from smt 
    mv temp_average_configuration_list.txt suggested_alps.txt 2>/dev/null
    
    if [ $WRITE_TO_DISK == 1 ]; then
        # remove compressed pickle files generated for this run
        rm *.pgz
    fi
done

