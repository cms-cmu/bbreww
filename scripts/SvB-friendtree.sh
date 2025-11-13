#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output/" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

# Create output directory
JOB="SvB_friendtree"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

# Modify the config file
display_section_header "Modifying config"
JOB_CONFIG=$OUTPUT_DIR/HHbbWW_make_friend_SvB.yml
sed -e "s|/srv/output/tmp/|$OUTPUT_DIR|" \
    bbreww/analysis/metadata/HHbbWW_make_friend_SvB.yml > $JOB_CONFIG
cat $JOB_CONFIG; echo

bash bbreww/scripts/run_processor.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --datasets "GluGluToHHTo2B2VLNu2J_kl_1p00 TTToSemiLeptonic  WtoLNu-2Jets_0J WtoLNu-2Jets_1J WtoLNu-2Jets_2J TbarWplustoLNu2Q TbarWplusto2L2Nu TWminustoLNu2Q TWminusto2L2Nu" \
    --year "2022_EE 2022_preEE 2023_preBPix 2023_BPix" \
    --output-filename "make_friend_SvB.coffea" \
    --output-subdir $JOB \
    --config $JOB_CONFIG \
    --no-test \
    --no-proxy \
    --condor
    # --additional-flags "--debug"
