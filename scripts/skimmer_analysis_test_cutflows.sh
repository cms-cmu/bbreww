#!/bin/bash

JOB_NAME="skimmer_analysis_test_cutflows"

# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output/" "$@")
if [ $? -ne 0 ]; then
    exit 1
fi

# Call the main analysis_test.sh script with Run3-specific parameters
bash bbww/scripts/run_cutflow.sh \
    --input-file "test.coffea" \
    --input-subdir "skimmer_analysis_test" \
    --output-base "$OUTPUT_BASE_DIR" \
    --output-filename "test_cutflow.yml" \
    --output-subdir "${JOB_NAME}" \
    --known-cutflow "bbww/tests/known_cutflow_skimmer_test.yml" 
