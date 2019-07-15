# run multiple times to generate multiple sets of images
export FEEDBACK_LOOP_RUNS=100

for counter in $(seq "$FEEDBACK_LOOP_RUNS")
do
    echo $RANDOM > random.txt
    
    # set the value of seed
    echo $counter > seed.txt
    
    # run feedback experiment
    ./do_feedback.sh

    # save results
    mv feedback_input_difference.txt "feedback_difference/"$counter"_feedback_input_difference.txt"
    mv feedback_output_difference.txt "feedback_difference/"$counter"_feedback_output_difference.txt"
    mv feedback_regression_parameter.txt "feedback_difference/"$counter"_feedback_regression_parameter.txt"
done

