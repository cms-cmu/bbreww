#!/bin/bash

JOB_NAME="analysis_test_mc"

# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output/" "$@")
if [ $? -ne 0 ]; then
    exit 1
fi

# Call the main analysis_test.sh script with Run3-specific parameters
bash bbreww/scripts/run_processor.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --processor "bbreww/analysis/processors/hh_bbww_processor.py" \
    --metadata "bbreww/metadata/skims_v3" \
    --config "bbreww/analysis/metadata/HHbbWW.yml" \
    --datasets "GluGluToHHTo2B2VLNu2J TTToSemiLeptonic" \
    --year "2022_EE" \
    --output-filename "test.coffea" \
    --output-subdir "${JOB_NAME}" 
