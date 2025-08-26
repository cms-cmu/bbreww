#!/bin/bash

# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output/" "$@")
if [ $? -ne 0 ]; then
    exit 1
fi

# Call the main analysis_test.sh script with Run3-specific parameters
bash bbww/scripts/run_processor.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --processor "bbww/analysis/processors/hh_bbww_processor.py" \
    --metadata "bbww/metadata/datasets.yml" \
    --config "bbww/analysis/metadata/HHbbWW.yml" \
    --datasets "GluGluToHHTo2B2VLNu2J TTToSemiLeptonic" \
    --year "2022_preEE 2022_EE" \
    --output-filename "output.coffea" \
    --output-subdir "full_run" \
    --no-test \
    --no-proxy
