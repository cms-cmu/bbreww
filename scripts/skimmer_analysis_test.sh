#!/bin/bash

# Source common functions
source "bbww/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "bbww/output/" "$@") || exit 1

[[ $(hostname) = *runner* ]] && OUTPUT_BASE_DIR="/builds/$CI_PROJECT_PATH/coffea4bees_framework/python/output"

INPUT_DIR="$OUTPUT_BASE_DIR/skimmer_test"
OUTPUT_DIR="$OUTPUT_BASE_DIR/skimmer_analysis_test"

display_section_header "Printing input yml file"
cat $INPUT_DIR/picoaod_datasets_GluGluToHHTo2B2VLNu2J_2022_preEE.yml

display_section_header "Modifying dataset file with skimmer ci output"
cmd=(python metadata/merge_yaml_datasets.py -m $INPUT_DIR/datasets.yml -f $INPUT_DIR/picoaod_datasets_GluGluToHHTo2B2VLNu2J_2022_preEE.yml -o $OUTPUT_DIR/datasets.yml)
echo "${cmd[@]}"
"${cmd[@]}"
cat $OUTPUT_DIR/datasets.yml


# Call the main analysis_test.sh script with Run3-specific parameters
bash bbww/scripts/analysis_test.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --processor "bbww/analysis/processors/hh_bbww_processor.py" \
    --metadata $OUTPUT_DIR/datasets.yml \
    --config "bbww/analysis/metadata/HHbbWW.yml" \
    --datasets "GluGluToHHTo2B2VLNu2J" \
    --year "2022_EE" \
    --output-filename "test.coffea" \
    --output-subdir "skimmer_analysis_test" \
    --no-test 


