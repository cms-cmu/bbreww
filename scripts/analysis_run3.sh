#!/bin/bash

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
    --metadata "bbreww/metadata/skims_v4" \
    --config "bbreww/analysis/metadata/HHbbWW.yml" \
    --datasets "data__EGamma data__SingleMuon GluGluToHHTo2B2VLNu2J_kl_1p00 TTToSemiLeptonic TTToHadronic TTTo2L2Nu WtoLNu-2Jets_0J WtoLNu-2Jets_1J WtoLNu-2Jets_2J TbarWplustoLNu2Q TbarWplusto2L2Nu TWminustoLNu2Q TWminusto2L2Nu" \
    --year "2022_preEE 2022_EE" \
    --output-filename "output.coffea" \
    --output-subdir "full_run" \
    --no-test \
    --no-proxy \
    --condor
#    data__EGamma data__SingleMuon TTToSemiLeptonic TTToHadronic TTTo2L2Nu WtoLNu-2Jets_0J WtoLNu-2Jets_1J WtoLNu-2Jets_2J TbarWplustoLNu2Q TbarWplusto2L2Nu TWminustoLNu2Q TWminusto2L2Nu TBbarQ TbarBQ TBbartoLplusNuBbar TbarBtoLminusNuB" \
