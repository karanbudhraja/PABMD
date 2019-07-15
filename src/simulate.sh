# load configuration
source configuration.sh

export SIMULATION_SCRIPT="sampling/"$ABM"/predict_"$ABM$SIMULATION_EXTENSION

# remove existing data
rm -f suggested_slps.txt
rm -rf sampling/$ABM/images_*

# for line in file
while read -r line
do
    # read the number of alps per line
    export ALPS_PER_LINE=$(echo $line | tr -cd ")" | wc -c)
    
    # remove all spaces in line
    # assume just one configuration per line, for now
    # fix this later by adding a loop
    export line=$(echo $line | sed "s/ //g")

    # simulate using suggested alps
    export SLP_CONFIGURATIONS=$($SIMULATION_TOOL $SIMULATION_SCRIPT $line)

    # in case of images, slp configurations are not returned
    # compute slp configuration from images
    if [ $IMAGES == 1 ]; then
	# all folders
	export SLP_CONFIGURATIONS="["
	
	# get the last image folder
	# this is the most recently created one
	# get as many lines as the number of alps in a row of suggested_alps.txt
	export SLP_IMAGES_FOLDERS=$(ls "sampling/"$ABM | sort | grep images_ | tail -n $ALPS_PER_LINE)

	for folder in $SLP_IMAGES_FOLDERS
	do
	    # for each folder
	    export SLP_CONFIGURATION=$(python sampling/utils/process_folder_images.py sampling/$ABM/$folder)
	    export CHARACTER_COUNT=$(echo $SLP_CONFIGURATIONS | wc -c)

	    if [ $CHARACTER_COUNT == 2 ]; then
		# skip first occurrence
		export SEPARATOR=""
	    else
		export SEPARATOR=", "
	    fi
	    
	    export SLP_CONFIGURATIONS=$SLP_CONFIGURATIONS$SEPARATOR$SLP_CONFIGURATION
	done

	# add closing bracket
	export SLP_CONFIGURATIONS=$SLP_CONFIGURATIONS"]"
    fi
    
    echo $SLP_CONFIGURATIONS >> suggested_slps.txt
done < suggested_alps.txt
