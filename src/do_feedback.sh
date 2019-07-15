# clean
rm -f feedback_*.txt
rm -f suggested_*.txt
rm -f queried_*.txt

# run multiple times to generate multiple sets of images
export ITERATIONS=100

for i in $(seq "$ITERATIONS")
do
    # store the current iteration value
    echo $i > feedback_iteration.txt
    
    # run experiment
    # simulate results
    ./lfd.sh && ./simulate.sh

    # observe data for slps
    # append to file
    cat suggested_slps.txt >> feedback_suggested_slps.txt
    cat queried_slps.txt >> feedback_queried_slps.txt
    cat regression_parameter.txt >> feedback_regression_parameter.txt
done

