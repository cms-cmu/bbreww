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
run_command python bbreww/plots/makePlots.py $INPUT_DIR/test.coffea --doTest -o $OUTPUT_DIR -m bbreww/plots/metadata/plotsAll.yml --modifiers bbreww/plots/metadata/plotModifiers.yml

### run this line to run locally for all plots
# python bbreww/plots/makePlots.py output/full_run/output.coffea -o output/analysis_plot_mc -m bbreww/plots/metadata/plotsAll.yml --modifiers bbreww/plots/metadata/plotModifiers.yml

display_section_header "Checking if pdf files exist"
ls $OUTPUT_DIR/Run3/preselection/flavor_sum/channel_sum/Hbb_mass.pdf
ls $OUTPUT_DIR/Run3/preselection/flavor_e/channel_sum/Hbb_mass.pdf
ls $OUTPUT_DIR/Run3/preselection/flavor_mu/channel_sum/Hbb_mass.pdf
#ls $OUTPUT_DIR/RunII/passPreSel/fourTag/SR/SvB_MA_ps_hh.pdf
#ls $OUTPUT_DIR/RunII/passPreSel/fourTag/SR_vs_SB/data/SvB_MA_ps.pdf
#ls $OUTPUT_DIR/RunII/passPreSel/fourTag/SR_vs_SB/HH4b/SvB_MA_ps.pdf
#ls $OUTPUT_DIR/RunII/passPreSel_vs_failSvB_vs_passSvB/fourTag/SR/data/v4j_mass.pdf
#ls $OUTPUT_DIR/RunII/passPreSel_vs_failSvB_vs_passSvB/fourTag/SR/HH4b/v4j_mass.pdf 
#ls $OUTPUT_DIR/RunII/passPreSel/fourTag/SR/data/quadJet_min_dr_close_vs_other_m.pdf 
#ls $OUTPUT_DIR/RunII/passPreSel/fourTag/SR/HH4b/quadJet_min_dr_close_vs_other_m.pdf
#ls $OUTPUT_DIR/RunII/passPreSel/threeTag/SR/Multijet/quadJet_min_dr_close_vs_other_m.pdf 


display_section_header "check making the plots from yaml "
run_command python src/plotting/plot_from_yaml.py --input_yaml \
        $OUTPUT_DIR/Run3/preselection/flavor_sum/channel_sum/Hbb_mass.yaml \
        $OUTPUT_DIR/Run3/preselection/flavor_e/channel_sum/Hbb_mass.yaml \
        $OUTPUT_DIR/Run3/preselection/flavor_mu/channel_sum/Hbb_mass.yaml \
        --out $OUTPUT_DIR/test_plots_from_yaml 
    
display_section_header "Checking if pdf files exist"
ls $OUTPUT_DIR/test_plots_from_yaml/Run3/preselection/flavor_sum/channel_sum/Hbb_mass.pdf
ls $OUTPUT_DIR/test_plots_from_yaml/Run3/preselection/flavor_e/channel_sum/Hbb_mass.pdf
ls $OUTPUT_DIR/test_plots_from_yaml/Run3/preselection/flavor_mu/channel_sum/Hbb_mass.pdf
# ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel/fourTag/SR/SvB_MA_ps_zh.pdf
# ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel/fourTag/SR/SvB_MA_ps_hh.pdf
# ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel/fourTag/SR_vs_SB/data/SvB_MA_ps.pdf
# ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel/fourTag/SR_vs_SB/HH4b/SvB_MA_ps.pdf
# ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel_vs_failSvB_vs_passSvB/fourTag/SR/data/v4j_mass.pdf
# ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel_vs_failSvB_vs_passSvB/fourTag/SR/HH4b/v4j_mass.pdf 
# ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel/fourTag/SR/data/quadJet_min_dr_close_vs_other_m.pdf 
# ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel/fourTag/SR/HH4b/quadJet_min_dr_close_vs_other_m.pdf
# ls $OUTPUT_DIR/test_plots_from_yaml/RunII/passPreSel/threeTag/SR/Multijet/quadJet_min_dr_close_vs_other_m.pdf 
