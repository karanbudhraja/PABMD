# load configuration
export USE_FILTER_DESCRIPTOR_SETTING=1
source configuration.sh

# filter configurations per demonstration
echo "=>filtering configurations"
python filter.py
