# this script does everything
# it is for reference of order for running scripts 

# sample alp, slp pairs
./sampling.sh

# run the framework to suggest several alps
# that would produce the demonstration slps
./lfd.sh

# simulate those alps
# to check the actual slps that they correspond to
./simulate.sh

# filter from the suggested alps
# use distance from demonstration slps
./filter.sh
