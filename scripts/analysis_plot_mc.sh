#!/bin/bash
# Source common functions
source "src/scripts/common.sh"


OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "$@")
if [ $? -ne 0 ]; then
    exit 1
fi


INPUT_DIR="${OUTPUT_BASE_DIR}/analysis_test_mc"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/analysis_plot_mc"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### ls input file"
ls $INPUT_DIR/
ls $INPUT_DIR/test.coffea

display_section_header "Running makePlots.py"
run_command python bbreww/plots/makePlots.py $INPUT_DIR/test.coffea --doTest -o $OUTPUT_DIR -m bbreww/plots/metadata/plotsTest.yml --modifiers bbreww/plots/metadata/plotModifiers.yml

### run this line to run locally for all plots
# python bbreww/plots/makePlots.py output/full_run/output.coffea -o output/analysis_plot_mc -m bbreww/plots/metadata/plotsAll.yml --modifiers bbreww/plots/metadata/plotModifiers.yml

# makePlots.py derives the year subdir from the year axis in the coffea
# (the CI test sample is a single year, e.g. '2022'), so resolve it
# dynamically instead of hardcoding a value that breaks when the test
# year changes.
YEAR_DIR=$(basename "$(ls -d $OUTPUT_DIR/*/nominal_4j2b 2>/dev/null | head -1 | xargs dirname)")
echo "Resolved plot year directory: ${YEAR_DIR}"

display_section_header "Checking if pdf files exist"
ls $OUTPUT_DIR/$YEAR_DIR/nominal_4j2b/flavor_sum/region_sum/HHbbWW/Hbb_mass.pdf
ls $OUTPUT_DIR/$YEAR_DIR/nominal_4j2b/flavor_sum/region_CR/HHbbWW/Hbb_mass.pdf
ls $OUTPUT_DIR/$YEAR_DIR/nominal_4j2b/flavor_sum/region_SR/HHbbWW/Hbb_mass.pdf


display_section_header "check making the plots from yaml "
run_command python src/plotting/plot_from_yaml.py --input_yaml \
        $OUTPUT_DIR/$YEAR_DIR/nominal_4j2b/flavor_sum/region_sum/HHbbWW/Hbb_mass.yaml \
        $OUTPUT_DIR/$YEAR_DIR/nominal_4j2b/flavor_sum/region_CR/HHbbWW/Hbb_mass.yaml \
        $OUTPUT_DIR/$YEAR_DIR/nominal_4j2b/flavor_sum/region_SR/HHbbWW/Hbb_mass.yaml \
        --out $OUTPUT_DIR/test_plots_from_yaml

display_section_header "Checking if pdf files exist"
ls $OUTPUT_DIR/test_plots_from_yaml/$YEAR_DIR/nominal_4j2b/flavor_sum/region_sum/HHbbWW/Hbb_mass.pdf
ls $OUTPUT_DIR/test_plots_from_yaml/$YEAR_DIR/nominal_4j2b/flavor_sum/region_CR/HHbbWW/Hbb_mass.pdf
ls $OUTPUT_DIR/test_plots_from_yaml/$YEAR_DIR/nominal_4j2b/flavor_sum/region_SR/HHbbWW/Hbb_mass.pdf
