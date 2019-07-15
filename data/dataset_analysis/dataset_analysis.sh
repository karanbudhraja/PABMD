# demonstration specification
export DEMONSTRATION_ID=1

# steps across dimensions
export DIM1_STEP=0.1
export DIM2_STEP=0.1

# clean old datasets
rm -f $DEMONSTRATION_ID/predictions_best/slp_dataset.txt
rm -f $DEMONSTRATION_ID/predictions_second_best/slp_dataset.txt
rm -f $DEMONSTRATION_ID/predictions_third_best/slp_dataset.txt
rm -f $DEMONSTRATION_ID/predictions_worst/slp_dataset.txt
rm -f $DEMONSTRATION_ID/predictions_second_worst/slp_dataset.txt
rm -f $DEMONSTRATION_ID/predictions_third_worst/slp_dataset.txt

# make a copy of the datasets
# uniform name for ease of use

cp $DEMONSTRATION_ID/predictions_best/slp_dataset* $DEMONSTRATION_ID/predictions_best/slp_dataset.txt
cp $DEMONSTRATION_ID/predictions_second_best/slp_dataset* $DEMONSTRATION_ID/predictions_second_best/slp_dataset.txt
cp $DEMONSTRATION_ID/predictions_third_best/slp_dataset* $DEMONSTRATION_ID/predictions_third_best/slp_dataset.txt
cp $DEMONSTRATION_ID/predictions_worst/slp_dataset* $DEMONSTRATION_ID/predictions_worst/slp_dataset.txt
cp $DEMONSTRATION_ID/predictions_second_worst/slp_dataset* $DEMONSTRATION_ID/predictions_second_worst/slp_dataset.txt
cp $DEMONSTRATION_ID/predictions_third_worst/slp_dataset* $DEMONSTRATION_ID/predictions_third_worst/slp_dataset.txt

# invoke python script
python dataset_analysis.py $DEMONSTRATION_ID $DIM1_STEP $DIM2_STEP
