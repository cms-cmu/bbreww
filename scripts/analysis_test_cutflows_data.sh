#!/bin/bash

# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "$@")
if [ $? -ne 0 ]; then
    exit 1
fi

# Call the main analysis_test.sh script with Run3-specific parameters
bash bbreww/scripts/run_cutflow.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --input-file "test.coffea" \
    --input-subdir "analysis_test_data" \
    --output-filename "test_cutflow_data.yml" \
    --output-subdir "analysis_test_cutflows_data" \
    --known-cutflow "bbreww/tests/known_cutflow_analysis_test_data.yml" 
