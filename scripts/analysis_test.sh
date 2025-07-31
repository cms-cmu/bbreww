#!/bin/bash

# Source common functions
source "bbww/scripts/common.sh"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --output-base DIR         Base output directory (default: bbww/output/)"
    echo "  --processor PATH          Path to processor file (default: bbww/analysis/processors/hh_bbww_processor.py)"
    echo "  --metadata PATH           Path to metadata file (default: bbww/metadata/datasets.yml)"
    echo "  --config PATH             Path to config file (default: bbww/analysis/metadata/HHbbWW.yml)"
    echo "  --triggers PATH           Path to triggers file (default: bbww/metadata/triggers_bbWW.yml)"
    echo "  --luminosities PATH       Path to luminosities file (default: bbww/metadata/luminosities_bbWW.yml)"
    echo "  --datasets \"DATASET1 DATASET2\"  Space-separated datasets (default: \"GluGluToHHTo2B2VLNu2J TTToSemiLeptonic\")"
    echo "  --year YEAR               Analysis year (default: UL18)"
    echo "  --output-filename FILE    Output filename (default: test.coffea)"
    echo "  --no-test                 Disable test mode"
    echo "  --no-proxy                Disable proxy setup"
    echo "  --output-subdir DIR       Output subdirectory (default: analysis_test)"
    echo "  --additional-flags FLAGS  Additional flags to pass to runner.py"
    echo "  --help                    Show this help message"
    exit 1
}

# Function to display configuration
display_config() {
    echo "############### Configuration"
    echo "Processor:        $PROCESSOR_PATH"
    echo "Metadata:         $METADATA_PATH"
    echo "Config:           $CONFIG_PATH"
    echo "Triggers:         $TRIGGERS_PATH"
    echo "Luminosities:     $LUMINOSITIES_PATH"
    echo "Datasets:         $DATASETS"
    echo "Year:             $YEAR"
    echo "Output filename:  $OUTPUT_FILENAME"
    echo "Test mode:        $([ -n "$TEST_MODE" ] && echo "enabled" || echo "disabled")"
    echo "Proxy setup:      $([ -n "$DO_PROXY" ] && echo "enabled" || echo "disabled")"
    echo "Output subdir:    $OUTPUT_SUBDIR"
    echo "Additional flags: ${ADDITIONAL_FLAGS:-"(none)"}"
    echo ""
}

# Default values
declare -A DEFAULTS=(
    ["OUTPUT_BASE"]="bbww/output/"
    ["PROCESSOR_PATH"]="bbww/analysis/processors/hh_bbww_processor.py"
    ["METADATA_PATH"]="bbww/metadata/datasets.yml"
    ["CONFIG_PATH"]="bbww/analysis/metadata/HHbbWW.yml"
    ["TRIGGERS_PATH"]="bbww/metadata/triggers_bbWW.yml"
    ["LUMINOSITIES_PATH"]="bbww/metadata/luminosities_bbWW.yml"
    ["DATASETS"]="GluGluToHHTo2B2VLNu2J TTToSemiLeptonic"
    ["YEAR"]="UL18"
    ["OUTPUT_FILENAME"]="test.coffea"
    ["TEST_MODE"]="-t"
    ["DO_PROXY"]="--do_proxy"
    ["OUTPUT_SUBDIR"]="analysis_test"
    ["ADDITIONAL_FLAGS"]=""
)

# Initialize variables with defaults
OUTPUT_BASE="${DEFAULTS[OUTPUT_BASE]}"
PROCESSOR_PATH="${DEFAULTS[PROCESSOR_PATH]}"
METADATA_PATH="${DEFAULTS[METADATA_PATH]}"
CONFIG_PATH="${DEFAULTS[CONFIG_PATH]}"
TRIGGERS_PATH="${DEFAULTS[TRIGGERS_PATH]}"
LUMINOSITIES_PATH="${DEFAULTS[LUMINOSITIES_PATH]}"
DATASETS="${DEFAULTS[DATASETS]}"
YEAR="${DEFAULTS[YEAR]}"
OUTPUT_FILENAME="${DEFAULTS[OUTPUT_FILENAME]}"
TEST_MODE="${DEFAULTS[TEST_MODE]}"
DO_PROXY="${DEFAULTS[DO_PROXY]}"
OUTPUT_SUBDIR="${DEFAULTS[OUTPUT_SUBDIR]}"
ADDITIONAL_FLAGS="${DEFAULTS[ADDITIONAL_FLAGS]}"

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
        --triggers)
            TRIGGERS_PATH="$2"
            shift 2
            ;;
        --luminosities)
            LUMINOSITIES_PATH="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
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
        --additional-flags)
            ADDITIONAL_FLAGS="$2"
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
    ["TRIGGERS_PATH"]="$TRIGGERS_PATH"
    ["LUMINOSITIES_PATH"]="$LUMINOSITIES_PATH"
    ["DATASETS"]="$DATASETS"
    ["YEAR"]="$YEAR"
    ["OUTPUT_FILENAME"]="$OUTPUT_FILENAME"
    ["TEST_MODE"]="$TEST_MODE"
    ["DO_PROXY"]="$DO_PROXY"
    ["OUTPUT_SUBDIR"]="$OUTPUT_SUBDIR"
    ["ADDITIONAL_FLAGS"]="$ADDITIONAL_FLAGS"
)

# Setup proxy if needed
setup_proxy "$DO_PROXY"

# Restore our configuration variables after setup
OUTPUT_BASE="${SAVED_VARS[OUTPUT_BASE]}"
PROCESSOR_PATH="${SAVED_VARS[PROCESSOR_PATH]}"
METADATA_PATH="${SAVED_VARS[METADATA_PATH]}"
CONFIG_PATH="${SAVED_VARS[CONFIG_PATH]}"
TRIGGERS_PATH="${SAVED_VARS[TRIGGERS_PATH]}"
LUMINOSITIES_PATH="${SAVED_VARS[LUMINOSITIES_PATH]}"
DATASETS="${SAVED_VARS[DATASETS]}"
YEAR="${SAVED_VARS[YEAR]}"
OUTPUT_FILENAME="${SAVED_VARS[OUTPUT_FILENAME]}"
TEST_MODE="${SAVED_VARS[TEST_MODE]}"
DO_PROXY="${SAVED_VARS[DO_PROXY]}"
OUTPUT_SUBDIR="${SAVED_VARS[OUTPUT_SUBDIR]}"
ADDITIONAL_FLAGS="${SAVED_VARS[ADDITIONAL_FLAGS]}"

# Display configuration
display_config

OUTPUT_DIR="${OUTPUT_BASE}/${OUTPUT_SUBDIR}/"
create_output_directory "$OUTPUT_DIR"

echo "############### Running test processor"
cmd=(python runner.py 
    -p "$PROCESSOR_PATH" 
    -m "$METADATA_PATH" 
    -c "$CONFIG_PATH" 
    --triggers "$TRIGGERS_PATH"
    --luminosities "$LUMINOSITIES_PATH"
    -d $DATASETS 
    -y "$YEAR" 
    -op "$OUTPUT_DIR" 
    -o "$OUTPUT_FILENAME" 
    $TEST_MODE 
    $ADDITIONAL_FLAGS
)
run_command "${cmd[@]}"

echo "############### Output files"
ls $OUTPUT_DIR