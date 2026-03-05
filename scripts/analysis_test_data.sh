#!/bin/bash
set -e  # Terminate script immediately if any command returns a non-zero exit code

JOB_NAME="analysis_test_data"

# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output/" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

# Call the main analysis_test.sh script with Run3-specific parameters
bash bbreww/scripts/run_processor.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --processor "bbreww/analysis/processors/hh_bbww_processor.py" \
    --metadata "bbreww/metadata/skims_v5" \
    --config "bbreww/analysis/metadata/HHbbWW.yml" \
    --datasets "data__SingleMuon0" \
    --year "2022_preEE" \
    --output-filename "test.coffea" \
    --output-subdir "${JOB_NAME}" 
