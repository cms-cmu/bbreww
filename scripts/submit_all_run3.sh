#!/bin/bash
# submit_all_run3.sh
#
# Opens a named screen session with one window per submission chunk.
# Each window runs its own ./run_container python runner.py --condor call
# and stays alive so you can watch progress and catch failures.
#
# To rerun a failed chunk, go into the window and hit up-arrow — each
# window runs a small rerun_<name>.sh script that stays in shell history.
# Rerun scripts are cleaned up automatically after the merge completes.
#
# Screen windows:
#   0: chunk1_2022_preEE  (GluGlu + TTToSemiLeptonic)
#   1: chunk1_2022_EE
#   2: chunk1_2023_preBPix
#   3: chunk1_2023_BPix
#   4: chunk2             (TTToHadronic + TTTo2L2Nu, all eras)
#   5: chunk3             (WtoLNu-2Jets, all eras)
#   6: chunk4             (TbarWplus + TWminus, all eras)
#   7: chunk5             (TBbar, all eras)
#   8: merge              (waits for all outputs, then merges)
#
# Usage:
#   bash bbreww/scripts/submit_all_run3.sh [--output-base DIR]
#   Default output base: output/
#
# Attach:         screen -r bbww_run3
# Switch windows: Ctrl-A N / Ctrl-A P   or   Ctrl-A " (list)
# Detach:         Ctrl-A D

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BARISTA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTPUT_BASE="output"
SESSION="bbww_run3"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-base) OUTPUT_BASE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--output-base DIR]"; exit 1 ;;
    esac
done

OUTPUT_DIR="${OUTPUT_BASE}/full_run"
mkdir -p "$OUTPUT_DIR"

# Shared runner.py flags (no -d / -y; set per chunk below)
COMMON="python runner.py \
    -p bbreww/analysis/processors/hh_bbww_processor.py \
    -m bbreww/metadata/skims_v5 \
    -c bbreww/analysis/metadata/HHbbWW.yml \
    --triggers bbreww/metadata/triggers_bbWW.yml \
    --luminosities bbreww/metadata/luminosities_bbWW.yml \
    --friends none \
    -op ${OUTPUT_DIR}/ \
    #--condor"

ALL_ERAS="2022_preEE 2022_EE 2023_preBPix 2023_BPix"
CHUNK1_DS="GluGluToHHTo2B2VLNu2J_kl_1p00 TTToSemiLeptonic"

# Temp dir for rerun scripts — cleaned up after merge
RERUN_DIR="$(mktemp -d /tmp/bbww_rerun_XXXXXX)"

# ---------------------------------------------------------------------------
# Write a small rerun script for each chunk so up-arrow works inside the window
# ---------------------------------------------------------------------------
write_rerun() {
    local name="$1"
    local cmd="$2"
    local f="${RERUN_DIR}/rerun_${name}.sh"
    printf '#!/bin/bash\ncd %s\n./run_container %s\n' "${BARISTA_DIR}" "${cmd}" > "$f"
    chmod +x "$f"
    echo "$f"
}

R1_preEE=$(write_rerun "chunk1_2022_preEE"   "${COMMON} -d ${CHUNK1_DS} -y 2022_preEE    -o output_chunk1_2022_preEE.coffea")
R1_EE=$(write_rerun    "chunk1_2022_EE"      "${COMMON} -d ${CHUNK1_DS} -y 2022_EE        -o output_chunk1_2022_EE.coffea")
R1_pre=$(write_rerun   "chunk1_2023_preBPix" "${COMMON} -d ${CHUNK1_DS} -y 2023_preBPix  -o output_chunk1_2023_preBPix.coffea")
R1_B=$(write_rerun     "chunk1_2023_BPix"    "${COMMON} -d ${CHUNK1_DS} -y 2023_BPix      -o output_chunk1_2023_BPix.coffea")
R2=$(write_rerun       "chunk2"  "${COMMON} -d TTToHadronic TTTo2L2Nu -y ${ALL_ERAS} -o output_chunk2.coffea")
R3=$(write_rerun       "chunk3"  "${COMMON} -d WtoLNu-2Jets_0J WtoLNu-2Jets_1J WtoLNu-2Jets_2J -y ${ALL_ERAS} -o output_chunk3.coffea")
R4=$(write_rerun       "chunk4"  "${COMMON} -d TbarWplustoLNu2Q TbarWplusto2L2Nu TWminustoLNu2Q TWminusto2L2Nu -y ${ALL_ERAS} -o output_chunk4.coffea")
R5=$(write_rerun       "chunk5"  "${COMMON} -d TBbarQ TbarBQ TBbartoLplusNuBbar TbarBtoLminusNuB -y ${ALL_ERAS} -o output_chunk5.coffea")

# ---------------------------------------------------------------------------
# Helper: open a screen window that runs the chunk's rerun script
# The rerun script is the first thing in shell history — just hit up-arrow
# ---------------------------------------------------------------------------
window_cmd() {
    local name="$1"
    local rerun="$2"
    local inner="cd ${BARISTA_DIR} && bash ${rerun}; echo; echo '>>> ${name} exited with code '\$?; exec bash"
    screen -S "$SESSION" -X screen -t "$name" bash -c "$inner"
}

# ---------------------------------------------------------------------------
# Kill any existing session with the same name, then create a fresh one
# ---------------------------------------------------------------------------
if screen -list | grep -q "\.${SESSION}[[:space:]]"; then
    echo "Killing existing screen session '${SESSION}'..."
    screen -S "$SESSION" -X quit || true
    sleep 1
fi

# Create session with first window
echo "Creating screen session '${SESSION}'..."
screen -dmS "$SESSION" -t "chunk1_2022_preEE" bash -c \
    "cd ${BARISTA_DIR} && bash ${R1_preEE}; echo; echo '>>> chunk1_2022_preEE exited with code '\$?; exec bash"

sleep 5  # give screen time to initialise

window_cmd "chunk1_2022_EE"      "$R1_EE"
sleep 5
window_cmd "chunk1_2023_preBPix" "$R1_pre"
sleep 5
window_cmd "chunk1_2023_BPix"    "$R1_B"
sleep 5
window_cmd "chunk2"              "$R2"
sleep 5
window_cmd "chunk3"              "$R3"
sleep 5
window_cmd "chunk4"              "$R4"
sleep 5
window_cmd "chunk5"              "$R5"

# ---------------------------------------------------------------------------
# Merge window — waits 30 min, polls every 5 min, merges, then cleans up
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

MERGE_SCRIPT="$(mktemp /tmp/bbww_merge_XXXXXX.sh)"
cat > "$MERGE_SCRIPT" << MERGEEOF
#!/bin/bash
cd "${BARISTA_DIR}"
log() { echo "[\$(date '+%H:%M:%S')] \$*"; }

EXPECTED=(
$(printf '    "%s"\n' "${EXPECTED_FILES[@]}")
)

RERUN_DIR="${RERUN_DIR}"

log "Rerun commands for all chunks:"
for rr in "\${RERUN_DIR}"/rerun_*.sh; do
    log "  bash \$rr"
done

log "Waiting 30 minutes before polling for output files..."
sleep 1800

log "Starting to poll for output files (every 5 minutes)..."
while true; do
    missing=()
    for f in "\${EXPECTED[@]}"; do
        [[ ! -f "\$f" ]] && missing+=("\$(basename "\$f")")
    done
    if [[ \${#missing[@]} -eq 0 ]]; then
        log "All files present."
        break
    fi
    log "Still waiting for: \${missing[*]}"
    for f in "\${missing[@]}"; do
        name="\${f%.coffea}"   # strip .coffea
        name="\${name#output_}" # strip output_ prefix
        log "  bash \${RERUN_DIR}/rerun_\${name}.sh"
    done
    sleep 300
done

log "Merging..."
./run_container python src/tools/merge_coffea_files.py \\
    -o "${OUTPUT_DIR}/output_merged.coffea" \\
    -f \${EXPECTED[@]}

log "Cleaning up rerun scripts from \${RERUN_DIR}..."
rm -rf "\${RERUN_DIR}"

log "Done. Merged output: ${OUTPUT_DIR}/output_merged.coffea"
exec bash
MERGEEOF
chmod +x "$MERGE_SCRIPT"

screen -S "$SESSION" -X screen -t "merge" bash "$MERGE_SCRIPT"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "Screen session '${SESSION}' is running with the following windows:"
echo "  0: chunk1_2022_preEE   — GluGlu + TTToSemiLeptonic (2022_preEE)"
echo "  1: chunk1_2022_EE      — GluGlu + TTToSemiLeptonic (2022_EE)"
echo "  2: chunk1_2023_preBPix — GluGlu + TTToSemiLeptonic (2023_preBPix)"
echo "  3: chunk1_2023_BPix    — GluGlu + TTToSemiLeptonic (2023_BPix)"
echo "  4: chunk2              — TTToHadronic + TTTo2L2Nu (all eras)"
echo "  5: chunk3              — WtoLNu-2Jets (all eras)"
echo "  6: chunk4              — TbarWplus + TWminus (all eras)"
echo "  7: chunk5              — TBbar (all eras)"
echo "  8: merge               — waiting 30 min then polling every 5 min"
echo ""
echo "To rerun a failed chunk: go to its window and hit up-arrow"
echo ""
echo "Attach:          screen -r ${SESSION}"
echo "Switch windows:  Ctrl-A N / Ctrl-A P   or   Ctrl-A \" (list)"
echo "Detach:          Ctrl-A D"
echo "Output dir:      ${OUTPUT_DIR}/"
