#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"bbww/output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}skimmer_test"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Changing metadata"
if [[ $(hostname) == *fnal* ]]; then
    BASE_PATH="/srv/python/${OUTPUT_DIR}/"
else
    BASE_PATH="/builds/${CI_PROJECT_PATH}/python/output/skimmer_test/"
fi

sed -e "s#base_path.*#base_path: ${BASE_PATH}#" \
    -e "s/\#max.*/maxchunks: 2/" \
    -e "s/\#test.*/test_files: 1/" \
    bbww/skimmer/metadata/HHbbWW.yml > $OUTPUT_DIR/HHbbWW.yml
cat $OUTPUT_DIR/HHbbWW.yml

nanoAOD_file="root://cmseos.fnal.gov//store/mc/RunIISummer20UL18NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/260000/DC7CD215-44B8-E34A-9E2A-E9B569B8B1DB.root"
sed "s|/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM.*|[ '${nanoAOD_file}' ]|" bbww/metadata/datasets.yml > "$OUTPUT_DIR/datasets.yml"

echo "############### Running test processor"
cmd=(python runner.py -s -p bbww/skimmer/processors/skimmer_bbWW.py -c $OUTPUT_DIR/HHbbWW.yml -y UL18 -d TTToSemiLeptonic -op $OUTPUT_DIR -o picoaod_datasets_TTToSemiLeptonic_UL18.yml -m $OUTPUT_DIR/datasets.yml  -t )
echo "${cmd[@]}"
"${cmd[@]}"

ls -R $OUTPUT_DIR
