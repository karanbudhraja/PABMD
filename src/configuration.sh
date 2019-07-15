#!/usr/bin/env bash
#######################################################################
# this is the only thing that should be edited when using this script #
#######################################################################

# abm being used
export ABM="flocking"
#export ABM="civil_violence"
#export ABM="forest_fire"
#export ABM="aids"
#export ABM="wolf_sheep"
#export ABM="eum"
#export ABM="schelling"
#export ABM="turbulence"

# use of disk to reduce ram consumption
export WRITE_TO_DISK=0

# use of demonstration images is off by default 
export IMAGES=0

# check if this abm depends on images
if [ $ABM == "flocking" ] || [ $ABM == "turbulence" ]; then
    export IMAGES=1
fi

# if feature reduction has been used
export REDUCED=0

# if image features are uniform across all stages of the framework
# used during filtering
export IMAGE_FEATURIZATION_HOMOGENEOUS=0

# descriptor setting if any
export DESCRIPTOR_SETTING=""

# check if this abm depends on images
# specify which descriptor to use
if [ $IMAGES == 1 ]; then
    # determine number of dependent variables
    #export DESCRIPTOR_SETTING="DENSITY"
    export DESCRIPTOR_SETTING="CONTOUR"
    #export DESCRIPTOR_SETTING="LBP"
    #export DESCRIPTOR_SETTING="MNIST"
    #export DESCRIPTOR_SETTING="VAE"
    #export DESCRIPTOR_SETTING="LSTM"
    #export DESCRIPTOR_SETTING="VGG"
    #export DESCRIPTOR_SETTING="RESNET50"
    #export DESCRIPTOR_SETTING="IMAGE_MATCH"
fi

# check for override in case of filtering
if [ ! -z $USE_FILTER_DESCRIPTOR_SETTING ]; then
    if [ $USE_FILTER_DESCRIPTOR_SETTING == 1 ]; then
        #export DESCRIPTOR_SETTING="RESNET50"
	#export DESCRIPTOR_SETTING="IMAGE_MATCH"
	export DESCRIPTOR_SETTING="CONTOUR"
    fi
fi

# distance metric used for filtering
# euclidean assumed by default
export DISTANCE_METHOD="EUCLIDEAN"
#export DISTANCE_METHOD="NORMALIZED"

############
# feedback #
############

export FEEDBACK=0
export LEARNING_RATE_FM_DATASET=0.0
export LEARNING_RATE_REGRESSION_PARAMETER=1.0
export REGRESSION_PARAMETER_LOWER_LIMIT=1
export REGRESSION_PARAMETER_UPPER_LIMIT=100

#####################
# dataset selection #
#####################

export DATASET_SELECTION=1

#########################
# refine configurations #
#########################

export CONFIGURATIONS_PRUNING=1
export CONFIGURATIONS_OUTLIER_DETECTION=1

#############################
# constants for this script #
#############################

# demonstration file
export DEMONSTRATION_FOLDER="../data/demonstrations/"$ABM

# configuration file
export CONFIGURATION_FILE="domainconfs/"$ABM".py"

# data file
# if image processing was used, append descriptor setting to processed filename
if [ $IMAGES == 1 ]; then
    export DESCRIPTOR_SETTING_LOWER_CASE=$(echo $DESCRIPTOR_SETTING  | tr "[:upper:]" "[:lower:]")
    export DATA_FILE="../data/domaindata/processed_"$ABM"_"$DESCRIPTOR_SETTING_LOWER_CASE

    if [ $REDUCED == 1 ]; then
	    export DATA_FILE=$DATA_FILE"_reduced"
    fi
    
    export DATA_FILE=$DATA_FILE".txt"
else
    export DATA_FILE="../data/domaindata/processed_"$ABM".txt"
fi

# scaling information
export SCALE_DATA_FILE=$(cat "map.py" | grep SCALE_DATA_FILE | head -n 1 | sed s/\"/\\n/g | sed -n 2p)

##########################################
# abm simulation tool and file extension #
# one entry required per abm             #
##########################################

if [ $ABM == "civil_violence" ] || [ $ABM == "forest_fire" ] || [ $ABM == "eum" ] || [ $ABM == "schelling" ]; then
    export SIMULATION_TOOL="python3"
    export SIMULATION_EXTENSION=".py"
elif [ $ABM == "flocking" ] || [ $ABM == "aids" ] || [ $ABM == "wolf_sheep" ] || [ $ABM == "turbulence" ]; then
    export SIMULATION_TOOL="java"
    export SIMULATION_EXTENSION=""
fi

############################################################
# image featurization details, if any                      #
# entry required for abms using images                     #
# one entry required per abm for nuber of dependent values #
############################################################

# default number of dependent values
export NUM_DEPENDENT=0

# check if this abm depends on images
if [ $IMAGES == 1 ]; then
    # determine number of dependent variables
    # based on image feature descriptors
    
    if [ "$DESCRIPTOR_SETTING" == "DENSITY" ]; then
	    export NUM_DEPENDENT=1
    elif [ "$DESCRIPTOR_SETTING" == "CONTOUR" ]; then
	    export NUM_DEPENDENT=2
    elif [ "$DESCRIPTOR_SETTING" == "LBP" ]; then
	    export NUM_DEPENDENT=128
    elif [ "$DESCRIPTOR_SETTING" == "MNIST" ]; then
	    export NUM_DEPENDENT=128
    elif [ "$DESCRIPTOR_SETTING" == "VAE" ]; then
	    export NUM_DEPENDENT=128
    elif [ "$DESCRIPTOR_SETTING" == "LSTM" ]; then
	    export NUM_DEPENDENT=128
    elif [ "$DESCRIPTOR_SETTING" == "VGG" ]; then
	    export NUM_DEPENDENT=1000
    elif [ "$DESCRIPTOR_SETTING" == "RESNET50" ]; then
	    export NUM_DEPENDENT=1000
    elif [ "$DESCRIPTOR_SETTING" == "IMAGE_MATCH" ]; then
	    export NUM_DEPENDENT=648
    fi

    if [ $REDUCED == 1 ]; then
	    export NUM_DEPENDENT=50
    fi
else
    # read from domainconfs
    export NUM_DEPENDENT=$(python -c "execd = {} ; exec(open(\""$CONFIGURATION_FILE"\").read(), execd) ; print execd[\"NUM_DEPENDENT\"]")
fi


