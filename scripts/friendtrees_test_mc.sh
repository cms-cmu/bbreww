#!/bin/bash

JOB_NAME="friendtrees_test_mc"

# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi
# Create output directory
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB_NAME
create_output_directory "$OUTPUT_DIR"

display_section_header "Modifying config"
JOB_CONFIG="${OUTPUT_DIR}/HHbbWW_friendtree.yml"
sed -e 's|\&friend_base.*|\&friend_base '$OUTPUT_DIR'|' \
    bbreww/analysis/metadata/HHbbWW_friendtree.yml > $JOB_CONFIG
cat $JOB_CONFIG; echo

display_section_header "Running processor"
bash bbreww/scripts/run_processor.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --processor "bbreww/analysis/processors/hh_bbww_processor.py" \
    --metadata "bbreww/metadata/skims_v4" \
    --config "$JOB_CONFIG" \
    --datasets "GluGluToHHTo2B2VLNu2J TTToSemiLeptonic" \
    --year "2022_EE" \
    --output-filename "test.coffea" \
    --output-subdir "${JOB_NAME}" 
