#!/bin/bash

# go to target script directory
cd ../src
source configuration.sh

# copy to demonstration folder for lfd use
cd ../data/demonstrations/$ABM
rm -rf *
cp -r ../../../app/matches .

# move to appropriate directory to run lfd
cd ../../../src

# run the framework to suggest several alps
# that would produce the demonstration slps
./lfd.sh

# simulate those alps
# to check the actual slps that they correspond to
./simulate.sh

# filter from the suggested alps
# use distance from demonstration slps
export USE_FILTER_DESCRIPTOR_SETTING=1
./filter.sh
