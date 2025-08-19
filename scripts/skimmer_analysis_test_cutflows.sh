#!/bin/bash

#!/bin/bash

# Source common functions
source "bbww/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output/" "$@")
if [ $? -ne 0 ]; then
    exit 1
fi

# Call the main analysis_test.sh script with Run3-specific parameters
bash bbww/scripts/analysis_test_cutflows.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --input-file "test.coffea" \
    --input-subdir "skimmer_analysis_test" \
    --output-filename "test_cutflow.yml" \
    --output-subdir "skimmer_analysis_test_cutflows" \
    --known-cutflow "bbww/tests/known_cutflow_skimmer_test.yml" 
