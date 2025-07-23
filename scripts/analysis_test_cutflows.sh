#!/bin/bash

# Check if the output directory is provided as an argument, otherwise use default
source scripts/set_initial_variables.sh --output ${1:-"bbww/output/"}

# Run the analysis step

# Define input and output paths
# Create output directory if it doesn't exist
OUTPUT_DIR="${DEFAULT_DIR}analysis_test_cutflow_job/"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi
INPUT_FILE="${DEFAULT_DIR}analysis_test_job/test.coffea"
OUTPUT_FILE="${OUTPUT_DIR}/test_cutflow.yml"

# Run the Python script
cmd=(python bbww/tests/dump_cutflow_to_yaml.py -i $INPUT_FILE -o $OUTPUT_FILE)
echo "Running: " "${cmd[@]}"
"${cmd[@]}"

cmd=(python bbww/tests/cutflow_unittest.py --input_file $OUTPUT_FILE --known_cutflow bbww/tests/known_cutflow_hh_bbww_processor.yml)
echo "Running: " "${cmd[@]}"
"${cmd[@]}"

ls -lh $OUTPUT_FILE

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Script run successfully. Have a good day!" 
else
    echo "Script failed."
fi
