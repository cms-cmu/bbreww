#!/bin/bash

JOB_NAME="skimmer_test"
CONFIG_PATH="bbreww/skimmer/metadata/HHbbWW.yml"
METADATA_PATH="bbreww/metadata/datasets.yml"

# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output/" "$@")
if [ $? -ne 0 ]; then
    exit 1
fi

OUTPUT_DIR="${OUTPUT_BASE_DIR}/${JOB_NAME}/"
create_output_directory "$OUTPUT_DIR"

display_section_header "Changing metadata"
BASE_PATH="/srv/${OUTPUT_DIR}/"
[[ $(hostname) = *runner* ]] && BASE_PATH="/builds/${CI_PROJECT_PATH}/${OUTPUT_DIR}/"

sed -e "s|base_path.*|base_path: ${BASE_PATH}|" \
    -e "s|#max.*|maxchunks: 2|" \
    -e "s|#test.*|test_files: 1|" \
    "$CONFIG_PATH" > "$OUTPUT_DIR/HHbbWW.yml"
cat "$OUTPUT_DIR/HHbbWW.yml"

nanoAOD_file="root://cmseos.fnal.gov//store/mc/Run3Summer22NanoAODv12/GluGlutoHHto2B2WtoLNu2Q_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v2/2550000/61bac832-a9e5-4106-8ecf-f620e5f4db6a.root"
# nanoAOD_file="root://cms-xrd-global.cern.ch//store/mc/Run3Summer22NanoAODv12/GluGlutoHHto2B2WtoLNu2Q_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v2/2550000/61bac832-a9e5-4106-8ecf-f620e5f4db6a.root"
sed "s|/GluGlutoHHto2B2WtoLNu2Q_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM|[ '${nanoAOD_file}' ]|" "$METADATA_PATH" > "$OUTPUT_DIR/datasets.yml"

bash bbww/scripts/run_processor.sh \
    --additional-flags "-s" \
    --output-base "$OUTPUT_BASE_DIR" \
    --processor "bbww/skimmer/processors/skimmer_bbWW.py" \
    --metadata "$OUTPUT_DIR/datasets.yml" \
    --config "$OUTPUT_DIR/HHbbWW.yml" \
    --datasets "GluGluToHHTo2B2VLNu2J" \
    --year "2022_preEE" \
    --output-filename "output.coffea" \
    --output-subdir "${JOB_NAME}" 
