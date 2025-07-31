#!/bin/bash

# Source common functions
source "bbww/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "bbww/output/" "$@")
if [ $? -ne 0 ]; then
    exit 1
fi

# Call the main analysis_test.sh script with Run3-specific parameters
bash bbww/scripts/analysis_test.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --processor "bbww/analysis/processors/hh_bbww_processor.py" \
    --metadata "bbww/metadata/datasets_run3.yml" \
    --config "bbww/analysis/metadata/HHbbWW.yml" \
    --datasets "GluGluToHHTo2B2VLNu2J TTToSemiLeptonic" \
    --year "2022_EE" \
    --output-filename "test.coffea" \
    --output-subdir "analysis_test_run3"
