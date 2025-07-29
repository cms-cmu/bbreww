#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --output-base DIR         Base output directory (default: bbww/output/)"
    echo "  --processor PATH          Path to processor file (default: bbww/analysis/processors/hh_bbww_processor.py)"
    echo "  --metadata PATH           Path to metadata file (default: bbww/metadata/datasets.yml)"
    echo "  --config PATH             Path to config file (default: bbww/analysis/metadata/HHbbWW.yml)"
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

# Default values
DEFAULT_OUTPUT_BASE="bbww/output/"
DEFAULT_PROCESSOR_PATH="bbww/analysis/processors/hh_bbww_processor.py"
DEFAULT_METADATA_PATH="bbww/metadata/datasets.yml"
DEFAULT_CONFIG_PATH="bbww/analysis/metadata/HHbbWW.yml"
DEFAULT_DATASETS="GluGluToHHTo2B2VLNu2J TTToSemiLeptonic"
DEFAULT_YEAR="UL18"
DEFAULT_OUTPUT_FILENAME="test.coffea"
DEFAULT_TEST_MODE="-t"
DEFAULT_DO_PROXY="--do_proxy"
DEFAULT_OUTPUT_SUBDIR="analysis_test"
DEFAULT_ADDITIONAL_FLAGS=""

# Initialize variables with defaults
OUTPUT_BASE="$DEFAULT_OUTPUT_BASE"
PROCESSOR_PATH="$DEFAULT_PROCESSOR_PATH"
METADATA_PATH="$DEFAULT_METADATA_PATH"
CONFIG_PATH="$DEFAULT_CONFIG_PATH"
DATASETS="$DEFAULT_DATASETS"
YEAR="$DEFAULT_YEAR"
OUTPUT_FILENAME="$DEFAULT_OUTPUT_FILENAME"
TEST_MODE="$DEFAULT_TEST_MODE"
DO_PROXY="$DEFAULT_DO_PROXY"
OUTPUT_SUBDIR="$DEFAULT_OUTPUT_SUBDIR"
ADDITIONAL_FLAGS="$DEFAULT_ADDITIONAL_FLAGS"

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

# Save our parsed values before sourcing the initial variables script
SAVED_PROCESSOR_PATH="$PROCESSOR_PATH"
SAVED_METADATA_PATH="$METADATA_PATH"
SAVED_CONFIG_PATH="$CONFIG_PATH"
SAVED_DATASETS="$DATASETS"
SAVED_YEAR="$YEAR"
SAVED_OUTPUT_FILENAME="$OUTPUT_FILENAME"
SAVED_TEST_MODE="$TEST_MODE"
SAVED_DO_PROXY="$DO_PROXY"
SAVED_OUTPUT_SUBDIR="$OUTPUT_SUBDIR"
SAVED_ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS"

source scripts/set_initial_variables.sh --output "$OUTPUT_BASE" $DO_PROXY

# Restore our configuration variables after sourcing
PROCESSOR_PATH="$SAVED_PROCESSOR_PATH"
METADATA_PATH="$SAVED_METADATA_PATH"
CONFIG_PATH="$SAVED_CONFIG_PATH"
DATASETS="$SAVED_DATASETS"
YEAR="$SAVED_YEAR"
OUTPUT_FILENAME="$SAVED_OUTPUT_FILENAME"
TEST_MODE="$SAVED_TEST_MODE"
DO_PROXY="$SAVED_DO_PROXY"
OUTPUT_SUBDIR="$SAVED_OUTPUT_SUBDIR"
ADDITIONAL_FLAGS="$SAVED_ADDITIONAL_FLAGS"

# Display configuration
echo "############### Configuration"
echo "Processor:        $PROCESSOR_PATH"
echo "Metadata:         $METADATA_PATH"
echo "Config:           $CONFIG_PATH"
echo "Datasets:         $DATASETS"
echo "Year:             $YEAR"
echo "Output filename:  $OUTPUT_FILENAME"
echo "Test mode:        $([ -n "$TEST_MODE" ] && echo "enabled" || echo "disabled")"
echo "Proxy setup:      $([ -n "$DO_PROXY" ] && echo "enabled" || echo "disabled")"
echo "Output subdir:    $OUTPUT_SUBDIR"
echo "Additional flags: ${ADDITIONAL_FLAGS:-"(none)"}"
echo ""

OUTPUT_DIR="${DEFAULT_DIR}/${OUTPUT_SUBDIR}/"
echo "############### Checking and creating output directory"
echo "Output directory: $OUTPUT_DIR"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running test processor"
cmd=(python runner.py 
    -p "$PROCESSOR_PATH" 
    -m "$METADATA_PATH" 
    -c "$CONFIG_PATH" 
    -d $DATASETS 
    -y "$YEAR" 
    -op "$OUTPUT_DIR" 
    -o "$OUTPUT_FILENAME" 
    $TEST_MODE 
    $ADDITIONAL_FLAGS
)
echo "Command: ${cmd[@]}"
"${cmd[@]}"

echo "############### Output files"
ls $OUTPUT_DIR