#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"bbww/output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}/analysis_test_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

# echo "############### Modifying config"
# sed -e "s|hist_cuts: .*|hist_cuts: [ passPreSel, passSvB, failSvB ]|" analysis/metadata/HH4b.yml > $OUTPUT_DIR/HH4b.yml
# cat $OUTPUT_DIR/HH4b.yml

echo "############### Running test processor"
cmd=(python runner.py -p bbww/analysis/processors/hh_bbww_processor.py -m bbww/metadata/datasets.yml -c bbww/analysis/metadata/HHbbWW.yml -d GluGluToHHTo2B2VLNu2J -y UL18 -op ${OUTPUT_DIR} -o test.coffea -t)
echo "${cmd[@]}"
"${cmd[@]}"

ls $OUTPUT_DIR