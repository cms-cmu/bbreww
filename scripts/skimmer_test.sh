#!/bin/bash

# Source common functions
source "bbww/scripts/common.sh"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --output-base DIR         Base output directory (default: bbww/output/)"
    echo "  --processor PATH          Path to processor file (default: bbww/skimmer/processors/skimmer_bbWW.py)"
    echo "  --metadata PATH           Path to metadata file (default: bbww/metadata/datasets.yml)"
    echo "  --config PATH             Path to config file (default: bbww/skimmer/metadata/HHbbWW.yml)"
    echo "  --dataset DATASET         Dataset to process (default: TTToSemiLeptonic)"
    echo "  --year YEAR               Analysis year (default: UL18)"
    echo "  --output-filename FILE    Output filename (default: picoaod_datasets_TTToSemiLeptonic_UL18.yml)"
    echo "  --no-test                 Disable test mode"
    echo "  --no-proxy                Disable proxy setup"
    echo "  --output-subdir DIR       Output subdirectory (default: skimmer_test)"
    echo "  --help                    Show this help message"
    exit 1
}

# Function to display configuration
display_config() {
    echo "############### Configuration"
    echo "Processor:        $PROCESSOR_PATH"
    echo "Metadata:         $METADATA_PATH"
    echo "Config:           $CONFIG_PATH"
    echo "Dataset:          $DATASET"
    echo "Year:             $YEAR"
    echo "Output filename:  $OUTPUT_FILENAME"
    echo "Test mode:        $([ -n "$TEST_MODE" ] && echo "enabled" || echo "disabled")"
    echo "Proxy setup:      $([ -n "$DO_PROXY" ] && echo "enabled" || echo "disabled")"
    echo "Output subdir:    $OUTPUT_SUBDIR"
    echo ""
}

# Default values
declare -A DEFAULTS=(
    ["OUTPUT_BASE"]="bbww/output/"
    ["PROCESSOR_PATH"]="bbww/skimmer/processors/skimmer_bbWW.py"
    ["METADATA_PATH"]="bbww/metadata/datasets_run3.yml"
    ["CONFIG_PATH"]="bbww/skimmer/metadata/HHbbWW.yml"
    ["DATASET"]="GluGluToHHTo2B2VLNu2J"
    ["YEAR"]="2022_preEE"
    ["OUTPUT_FILENAME"]="picoaod_datasets_GluGluToHHTo2B2VLNu2J_2022_preEE.yml"
    ["TEST_MODE"]="-t"
    ["DO_PROXY"]="--do_proxy"
    ["OUTPUT_SUBDIR"]="skimmer_test"
)

# Initialize variables with defaults
OUTPUT_BASE="${DEFAULTS[OUTPUT_BASE]}"
PROCESSOR_PATH="${DEFAULTS[PROCESSOR_PATH]}"
METADATA_PATH="${DEFAULTS[METADATA_PATH]}"
CONFIG_PATH="${DEFAULTS[CONFIG_PATH]}"
DATASET="${DEFAULTS[DATASET]}"
YEAR="${DEFAULTS[YEAR]}"
OUTPUT_FILENAME="${DEFAULTS[OUTPUT_FILENAME]}"
TEST_MODE="${DEFAULTS[TEST_MODE]}"
DO_PROXY="${DEFAULTS[DO_PROXY]}"
OUTPUT_SUBDIR="${DEFAULTS[OUTPUT_SUBDIR]}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --processor)
            PROCESSOR_PATH="$2"
            shift 2
            ;;
        --metadata)
            METADATA_PATH="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --year)
            YEAR="$2"
            shift 2
            ;;
        --output-filename)
            OUTPUT_FILENAME="$2"
            shift 2
            ;;
        --no-test)
            TEST_MODE=""
            shift
            ;;
        --no-proxy)
            DO_PROXY=""
            shift
            ;;
        --output-subdir)
            OUTPUT_SUBDIR="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Save our parsed values before setting up environment
declare -A SAVED_VARS=(
    ["OUTPUT_BASE"]="$OUTPUT_BASE"
    ["PROCESSOR_PATH"]="$PROCESSOR_PATH"
    ["METADATA_PATH"]="$METADATA_PATH"
    ["CONFIG_PATH"]="$CONFIG_PATH"
    ["DATASET"]="$DATASET"
    ["YEAR"]="$YEAR"
    ["OUTPUT_FILENAME"]="$OUTPUT_FILENAME"
    ["TEST_MODE"]="$TEST_MODE"
    ["DO_PROXY"]="$DO_PROXY"
    ["OUTPUT_SUBDIR"]="$OUTPUT_SUBDIR"
)

# Setup proxy if needed
setup_proxy "$DO_PROXY"

# Restore our configuration variables after setup
OUTPUT_BASE="${SAVED_VARS[OUTPUT_BASE]}"
PROCESSOR_PATH="${SAVED_VARS[PROCESSOR_PATH]}"
METADATA_PATH="${SAVED_VARS[METADATA_PATH]}"
CONFIG_PATH="${SAVED_VARS[CONFIG_PATH]}"
DATASET="${SAVED_VARS[DATASET]}"
YEAR="${SAVED_VARS[YEAR]}"
OUTPUT_FILENAME="${SAVED_VARS[OUTPUT_FILENAME]}"
TEST_MODE="${SAVED_VARS[TEST_MODE]}"
DO_PROXY="${SAVED_VARS[DO_PROXY]}"
OUTPUT_SUBDIR="${SAVED_VARS[OUTPUT_SUBDIR]}"

# Display configuration
display_config

OUTPUT_DIR="${OUTPUT_BASE}/${OUTPUT_SUBDIR}/"
create_output_directory "$OUTPUT_DIR"

echo "############### Changing metadata"
if [[ $(hostname) == *fnal* ]]; then
    BASE_PATH="/srv/python/${OUTPUT_DIR}/"
else
    BASE_PATH="/builds/${CI_PROJECT_PATH}/bbww/coffea4bees_framework/python/output/${OUTPUT_SUBDIR}/"
fi

sed -e "s#base_path.*#base_path: ${BASE_PATH}#" \
    -e "s/\#max.*/maxchunks: 2/" \
    -e "s/\#test.*/test_files: 1/" \
    "$CONFIG_PATH" > "$OUTPUT_DIR/HHbbWW.yml"
cat "$OUTPUT_DIR/HHbbWW.yml"

nanoAOD_file="root://cmseos.fnal.gov//store/mc/Run3Summer22NanoAODv12/GluGlutoHHto2B2WtoLNu2Q_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v2/2550000/61bac832-a9e5-4106-8ecf-f620e5f4db6a.root"
# nanoAOD_file="root://cms-xrd-global.cern.ch//store/mc/Run3Summer22NanoAODv12/GluGlutoHHto2B2WtoLNu2Q_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v2/2550000/61bac832-a9e5-4106-8ecf-f620e5f4db6a.root"
sed "s|/GluGlutoHHto2B2WtoLNu2Q_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM|[ '${nanoAOD_file}' ]|" "$METADATA_PATH" > "$OUTPUT_DIR/datasets.yml"

echo "############### Running test processor"
cmd=(python runner.py 
    -s 
    -p "$PROCESSOR_PATH" 
    -c "$OUTPUT_DIR/HHbbWW.yml" 
    -y "$YEAR" 
    -d "$DATASET" 
    -op "$OUTPUT_DIR" 
    -o "$OUTPUT_FILENAME" 
    -m "$OUTPUT_DIR/datasets.yml"  
    $TEST_MODE
)
run_command "${cmd[@]}"

echo "############### Output files"
ls -R "$OUTPUT_DIR"
