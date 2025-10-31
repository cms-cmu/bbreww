#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

INPUT_DIR="${OUTPUT_BASE_DIR}/analysis_test_mc"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/analysis_iplot_mc"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### ls input file"
ls $INPUT_DIR/
ls $INPUT_DIR/test.coffea


display_section_header "Running iPlot test"
run_command python bbreww/plots/tests/iPlot_test.py \
    --inputFile $INPUT_DIR/test.coffea
