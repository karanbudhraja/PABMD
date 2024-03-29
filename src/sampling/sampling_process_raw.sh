# copy raw data file if needed
if [ $IMAGES == 1 ]; then
    # raw data file is present in non-repository directory
    cp ../../../_swarm-lfd-data/$ABM/$ABM.txt .
fi

# separate head from rest of data
head -n 1 $ABM.txt > header.txt
tail -n +2 $ABM.txt > $ABM"_.txt"
mv $ABM"_.txt" $ABM.txt

# process images if any
# check if this abm depends on images
if [ $IMAGES == 1 ]; then
    # images and raw data file are present in non-repository directory
    mv ../../../_swarm-lfd-data/$ABM/images $ABM/
    python utils/process_training_images.py
    mv $ABM/images ../../../_swarm-lfd-data/$ABM/
fi

# extract independent component
# generate dependent component
# combine results
# overwrite header file
export INDEPENDENT_HEADER="i"
export DEPENDENT_HEADER="d"
export INDEPENDENT_HEADER_STRING=$(sed "s/[^"$INDEPENDENT_HEADER"']//g" <<< cat header.txt)
export DEPENDENT_HEADER_STRING=$(printf $DEPENDENT_HEADER {1..$NUM_DEPENDENT})
echo $INDEPENDENT_HEADER_STRING$DEPENDENT_HEADER_STRING > header.txt

# prepare data for amf input
echo "==>preparing data for amf"
python ../map.py $ABM.txt $ABM > processed_$ABM.txt
cat header.txt processed_$ABM.txt > temp.txt
mv temp.txt processed_$ABM.txt
rm header.txt
mv processed_$ABM.txt ../../data/domaindata

# if image processing was used, append descriptor setting to processed filename
if [ $IMAGES == 1 ]; then
    # generate lower case descriptor name
    export DESCRIPTOR_SETTING_LOWER=$(echo "$DESCRIPTOR_SETTING" | tr '[:upper:]' '[:lower:]')
    mv ../../data/domaindata/processed_$ABM.txt ../../data/domaindata/processed_$ABM"_"$DESCRIPTOR_SETTING_LOWER.txt
fi

# this file is no longer needed for computation
# the forward and reverse mappings are built from the processed file
echo "==>cleaning"
rm $ABM.txt

# move scale data to the abm folder
# assume only one scale data file exists here right now
mv scale_data_$ABM*.txt $ABM
