# load configuration
source configuration.sh
export NUM_INDEPENDENT=$(python -c "execd = {} ; exec(open(\""$CONFIGURATION_FILE"\").read(), execd) ; print execd[\"NUM_INDEPENDENT\"]")

# clean
if [ $IMAGES == 1 ]; then
    rm -f "nn_evaluation_"$ABM"_"$DESCRIPTOR_SETTING_LOWER_CASE"_mse.txt"
else
    rm -f "nn_evaluation_"$ABM"_mse.txt"
fi

# evaluate
export K=10
#export K=1
export TESTS_PER_FOLD=1

for i in $(seq 0 $(expr "$K" - 1))
do
    for j in $(seq 0 $(expr "$TESTS_PER_FOLD" - 1))
    do
	export SPLIT_NUMBER=$i
	
	# train rm network
	#python3 neural_network.py --abm $ABM --numIndependent $NUM_INDEPENDENT --numDependent $NUM_DEPENDENT --fmOnly 0 --rmOnly 1 --splitNumber $SPLIT_NUMBER --rmType mlp
	python3 neural_network.py --abm $ABM --numIndependent $NUM_INDEPENDENT --numDependent $NUM_DEPENDENT --fmOnly 0 --rmOnly 0 --splitNumber $SPLIT_NUMBER
    done
done
