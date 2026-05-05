#!/bin/bash
# submit_all_run3_falcon.sh
#
# Submits one sbatch job per chunk using barista/software/slurm/slurm_processor.conf,
# then submits a merge job that runs only after all chunks succeed.
#
# Logs land in OUTPUT_DIR/slurm_logs/<jobname>_<jobid>.out
#
# Usage:
#   bash bbreww/scripts/submit_all_run3_falcon.sh [--output-base DIR]
#   Default output base: output/

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BARISTA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SLURM_CONF="${BARISTA_DIR}/software/slurm/slurm_processor.conf"

OUTPUT_BASE="output"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-base) OUTPUT_BASE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--output-base DIR]"; exit 1 ;;
    esac
done

OUTPUT_DIR="${OUTPUT_BASE}/full_run"
LOG_DIR="${OUTPUT_DIR}/slurm_logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

ALL_ERAS="2022_preEE 2022_EE 2023_preBPix 2023_BPix"
CHUNK1_DS="GluGluToHHTo2B2VLNu2J_kl_1p00 TTToSemiLeptonic"

COMMON="python runner.py \
    -p bbreww/analysis/processors/hh_bbww_processor.py \
    -m bbreww/metadata/skims_v5 \
    -c bbreww/analysis/metadata/HHbbWW.yml \
    --triggers bbreww/metadata/triggers_bbWW.yml \
    --luminosities bbreww/metadata/luminosities_bbWW.yml \
    --friends none \
    -op ${OUTPUT_DIR}/"

# ---------------------------------------------------------------------------
# Helper: submit one chunk as an sbatch job, print the job ID
# ---------------------------------------------------------------------------
submit_chunk() {
    local name="$1"
    local runner_args="$2"

    sbatch --parsable \
        $(sed '/^#SBATCH /!d; s/^#SBATCH //' "$SLURM_CONF" | tr '\n' ' ') \
        --job-name="bbww_${name}" \
        --output="${LOG_DIR}/bbww_${name}_%j.out" \
        --error="${LOG_DIR}/bbww_${name}_%j.out" \
        --wrap="cd ${BARISTA_DIR} && ./run_container ${COMMON} ${runner_args}"
}

# ---------------------------------------------------------------------------
# Submit chunks
# ---------------------------------------------------------------------------
JID1_preEE=$(submit_chunk "chunk1_2022_preEE"   "-d ${CHUNK1_DS} -y 2022_preEE    -o output_chunk1_2022_preEE.coffea")
echo "Submitted chunk1_2022_preEE   — job ${JID1_preEE}"

JID1_EE=$(submit_chunk    "chunk1_2022_EE"      "-d ${CHUNK1_DS} -y 2022_EE        -o output_chunk1_2022_EE.coffea")
echo "Submitted chunk1_2022_EE      — job ${JID1_EE}"

JID1_pre=$(submit_chunk   "chunk1_2023_preBPix" "-d ${CHUNK1_DS} -y 2023_preBPix  -o output_chunk1_2023_preBPix.coffea")
echo "Submitted chunk1_2023_preBPix — job ${JID1_pre}"

JID1_B=$(submit_chunk     "chunk1_2023_BPix"    "-d ${CHUNK1_DS} -y 2023_BPix      -o output_chunk1_2023_BPix.coffea")
echo "Submitted chunk1_2023_BPix    — job ${JID1_B}"

JID2=$(submit_chunk "chunk2" "-d TTToHadronic TTTo2L2Nu -y ${ALL_ERAS} -o output_chunk2.coffea")
echo "Submitted chunk2              — job ${JID2}"

JID3=$(submit_chunk "chunk3" "-d WtoLNu-2Jets_0J WtoLNu-2Jets_1J WtoLNu-2Jets_2J -y ${ALL_ERAS} -o output_chunk3.coffea")
echo "Submitted chunk3              — job ${JID3}"

JID4=$(submit_chunk "chunk4" "-d TbarWplustoLNu2Q TbarWplusto2L2Nu TWminustoLNu2Q TWminusto2L2Nu -y ${ALL_ERAS} -o output_chunk4.coffea")
echo "Submitted chunk4              — job ${JID4}"

JID5=$(submit_chunk "chunk5" "-d TBbarQ TbarBQ TBbartoLplusNuBbar TbarBtoLminusNuB -y ${ALL_ERAS} -o output_chunk5.coffea")
echo "Submitted chunk5              — job ${JID5}"

# ---------------------------------------------------------------------------
# Merge job — runs only if all chunks succeed
# ---------------------------------------------------------------------------
EXPECTED_FILES=(
    "${OUTPUT_DIR}/output_chunk1_2022_preEE.coffea"
    "${OUTPUT_DIR}/output_chunk1_2022_EE.coffea"
    "${OUTPUT_DIR}/output_chunk1_2023_preBPix.coffea"
    "${OUTPUT_DIR}/output_chunk1_2023_BPix.coffea"
    "${OUTPUT_DIR}/output_chunk2.coffea"
    "${OUTPUT_DIR}/output_chunk3.coffea"
    "${OUTPUT_DIR}/output_chunk4.coffea"
    "${OUTPUT_DIR}/output_chunk5.coffea"
)

MERGE_WRAP="cd ${BARISTA_DIR} && \
    missing=(); for f in ${EXPECTED_FILES[*]}; do [[ -f \"\$f\" ]] || missing+=(\"\$f\"); done; \
    if [[ \${#missing[@]} -gt 0 ]]; then echo \"Missing chunk outputs:\"; printf '  %s\n' \"\${missing[@]}\"; exit 1; fi && \
    ./run_container python src/tools/merge_coffea_files.py \
    -o ${OUTPUT_DIR}/output_merged.coffea \
    -f ${EXPECTED_FILES[*]}"

DEP="${JID1_preEE}:${JID1_EE}:${JID1_pre}:${JID1_B}:${JID2}:${JID3}:${JID4}:${JID5}"

JID_MERGE=$(sbatch --parsable \
    $(sed '/^#SBATCH /!d; s/^#SBATCH //' "$SLURM_CONF" | tr '\n' ' ') \
    --job-name="bbww_merge" \
    --output="${LOG_DIR}/bbww_merge_%j.out" \
    --error="${LOG_DIR}/bbww_merge_%j.out" \
    --dependency="afterany:${DEP}" \
    --wrap="bash -c $(printf '%q' "$MERGE_WRAP")")
echo "Submitted merge               — job ${JID_MERGE} (depends on all chunks)"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/bbww_<chunk>_<jobid>.out"
echo ""
echo "Output dir: ${OUTPUT_DIR}/"
echo "Logs dir:   ${LOG_DIR}/"
