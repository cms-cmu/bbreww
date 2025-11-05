#!/bin/bash

JOB_NAME="skimmer_analysis_test"

# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output/" "$@") || exit 1

[[ $(hostname) = *runner* ]] && OUTPUT_BASE_DIR="/builds/$CI_PROJECT_PATH/output"

INPUT_DIR="$OUTPUT_BASE_DIR/skimmer_test"
OUTPUT_DIR="$OUTPUT_BASE_DIR/${JOB_NAME}"
create_output_directory "$OUTPUT_DIR"

display_section_header "Printing input yml file"
cat $INPUT_DIR/picoaod_datasets.yml

display_section_header "Modifying dataset file with skimmer ci output"
run_command python src/tools/merge_yaml_datasets.py \
    -m $INPUT_DIR/datasets.yml \
    -f $INPUT_DIR/picoaod_datasets.yml \
    -o $OUTPUT_DIR/datasets.yml
cat $OUTPUT_DIR/datasets.yml

# Call the main analysis_test.sh script with Run3-specific parameters
bash bbreww/scripts/run_processor.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --processor "bbreww/analysis/processors/hh_bbww_processor.py" \
    --metadata $OUTPUT_DIR/datasets.yml \
    --config "bbreww/analysis/metadata/HHbbWW.yml" \
    --datasets "GluGluToHHTo2B2VLNu2J_kl_1p00" \
    --year "2022_EE" \
    --output-filename "test.coffea" \
    --output-subdir "${JOB_NAME}" \
    --no-test 


