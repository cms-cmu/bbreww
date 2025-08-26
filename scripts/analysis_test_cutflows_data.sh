#!/bin/bash

#!/bin/bash

# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "bbww/output" "$@")
if [ $? -ne 0 ]; then
    exit 1
fi

# Call the main analysis_test.sh script with Run3-specific parameters
bash bbww/scripts/run_cutflow.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --input-file "test_data.coffea" \
    --input-subdir "analysis_test_run3" \
    --output-filename "test_cutflow_data.yml" \
    --output-subdir "analysis_test_cutflows_run3" \
    --known-cutflow "bbww/tests/known_cutflow_data_hh_bbww_processor.yml" 
