#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"bbww/output/"} --do_proxy

INPUT_DIR="${DEFAULT_DIR}skimmer_test"
OUTPUT_DIR="${DEFAULT_DIR}skimmer_analysis_test"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

if [[ $(hostname) = *fnal* ]]; then
    echo "No changing files"
else
    echo "############### Modifying previous dataset file (to read local files)"
    sed -i "s|\/builds/$CI_PROJECT_PATH\/python\/||g" $INPUT_DIR/picoaod_datasets_TTToSemiLeptonic_UL18.yml
    cat $INPUT_DIR/picoaod_datasets_TTToSemiLeptonic_UL18.yml
fi
echo "############### Modifying dataset file with skimmer ci output"
cmd=(python metadata/merge_yaml_datasets.py -m $INPUT_DIR/datasets.yml -f $INPUT_DIR/picoaod_datasets_TTToSemiLeptonic_UL18.yml -o $OUTPUT_DIR/datasets_HHbbWW.yml)
echo "${cmd[@]}"
"${cmd[@]}"
cat $OUTPUT_DIR/datasets_HHbbWW.yml

echo "############### Running test processor"
cmd=(python runner.py -o test_skimmer.coffea -d TTToSemiLeptonic -p bbww/analysis/processors/hh_bbww_processor.py -y UL18 -op $OUTPUT_DIR -c bbww/analysis/metadata/HHbbWW.yml -m $OUTPUT_DIR/datasets_HHbbWW.yml)
echo "${cmd[@]}"
"${cmd[@]}"
