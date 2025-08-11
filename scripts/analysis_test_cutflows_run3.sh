#!/bin/bash

#!/bin/bash

# Source common functions
source "bbww/scripts/common.sh"

if [ $? -ne 0 ]; then
    exit 1
fi

# Call the main analysis_test.sh script with Run3-specific parameters
bash bbww/scripts/analysis_test_cutflows.sh \
    --input-file "test.coffea" \
    --input-subdir "analysis_test_run3" \
    --output-filename "test_cutflow.yml" \
    --output-subdir "analysis_test_cutflows_run3" \
    --known-cutflow "bbww/tests/known_cutflow_hh_bbww_processor_run3.yml" 
