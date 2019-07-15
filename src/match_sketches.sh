# use neural network image features for matching
export USE_LSH_DESCRIPTOR_SETTING=1

# load configuration
source configuration.sh

# execute script
python match_sketches.py ../data/demonstrations_all/$ABM/sketched .
