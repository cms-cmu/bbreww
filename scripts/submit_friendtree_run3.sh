#!/bin/bash
# submit_friendtree_run3.sh
#
# Opens a named screen session with one window per submission chunk.
# Intermediate per-chunk outputs (.coffea + .json sidecars) go to
# output/friendtrees/<runid>/ so reruns never overwrite a previous run's
# files and the per-chunk JSONs are preserved alongside the merged output.
# The merge window polls for all per-chunk JSON metafiles, merges them
# with merge_friend_meta.py, and writes the final JSON to output/friendtrees/.
# Per-chunk JSONs are NOT cleaned up after merge.
#
# Modes (selected via --mode):
#   mc   — MC samples only:
#            chunk1_<era>  (GluGlu kl×4 + TTToSemiLeptonic, one per era)
#            chunk2        (TTToHadronic + TTTo2L2Nu, all eras)
#            chunk3        (WtoLNu-2Jets, all eras)
#            chunk4        (TbarWplus + TWminus, all eras)
#            chunk5        (TBbar, all eras)
#   data — Data samples only:
#            data_egamma      (data__EGamma, all eras)
#            data_singlemuon  (data__SingleMuon, all eras)
#   full — MC + Data
#
# Usage:
#   bash bbreww/scripts/submit_friendtree_run3.sh --mode <mc|data|full>
#                    [-o NAME] [--output-base DIR]
#                    [--chunk1 "ds1 ds2 ..."] [--chunk2 "ds1 ds2 ..."]
#                    [--chunk3 "ds1 ds2 ..."] [--chunk4 "ds1 ds2 ..."]
#                    [--chunk5 "ds1 ds2 ..."] [--eras "era1 era2 ..."]
#   --mode           required: mc, data, or full
#   -o NAME          base name for the merged output (default: friendtree)
#                    .json extension is added automatically
#   --output-base    directory for the final merged output (default: output/friendtrees)
#   --chunk1..5      space-separated list of datasets for that chunk
#   --eras           space-separated list of eras for all-era chunks (default: all 4 Run3 eras)
#
# Attach:         screen -r bbww_friendtree
# Switch windows: Ctrl-A N / Ctrl-A P   or   Ctrl-A " (list)
# Detach:         Ctrl-A D

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BARISTA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$BARISTA_DIR"

OUTPUT_NAME="friendtree"
OUTPUT_BASE="output/friendtrees"
SESSION="bbww_friendtree"
MODE=""

ALL_ERAS="2022_preEE 2022_EE 2023_preBPix 2023_BPix"
CHUNK1_DS="GluGluToHHTo2B2VLNu2J_kl_0p00 GluGluToHHTo2B2VLNu2J_kl_1p00 GluGluToHHTo2B2VLNu2J_kl_2p45 GluGluToHHTo2B2VLNu2J_kl_5p00 GluGlutoHHto2B2Zto2L2Q_kl_1p00 GluGlutoHHto2B2Tau_kl_1p00 TTToSemiLeptonic"
CHUNK2_DS="TTToHadronic TTTo2L2Nu"
CHUNK3_DS="WtoLNu-2Jets_0J WtoLNu-2Jets_1J WtoLNu-2Jets_2J"
CHUNK4_DS="TbarWplustoLNu2Q TbarWplusto2L2Nu TWminustoLNu2Q TWminusto2L2Nu"
CHUNK5_DS="TBbarQ TbarBQ TBbartoLplusNuBbar TbarBtoLminusNuB"
DATA_EG_DS="data__EGamma"
DATA_MU_DS="data__SingleMuon"

usage() {
    cat <<EOF
Usage: $0 --mode <mc|data|full> [-o NAME] [--output-base DIR]
       [--eras "era1 era2 ..."] [--chunk1..5 "ds1 ds2 ..."]

Modes:
  mc    Run MC chunks only (chunk1_<era>, chunk2..5)
  data  Run the data chunks only (data_egamma + data_singlemuon, all eras)
  full  Run MC and Data chunks together

Optional:
  -o NAME           Base name for merged output (default: friendtree)
  --output-base DIR Directory for final merged output (default: output/friendtrees)
  --chunk1..5       Override datasets for a specific MC chunk
  --eras            Override the list of eras for all-era chunks
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)        MODE="$2";       shift 2 ;;
        -o)            OUTPUT_NAME="$2"; shift 2 ;;
        --output-base) OUTPUT_BASE="$2"; shift 2 ;;
        --eras)        ALL_ERAS="$2";    shift 2 ;;
        --chunk1)      CHUNK1_DS="$2";   shift 2 ;;
        --chunk2)      CHUNK2_DS="$2";   shift 2 ;;
        --chunk3)      CHUNK3_DS="$2";   shift 2 ;;
        --chunk4)      CHUNK4_DS="$2";   shift 2 ;;
        --chunk5)      CHUNK5_DS="$2";   shift 2 ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "Unknown option: $1"; echo; usage; exit 1 ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "ERROR: --mode is required."
    echo
    usage
    exit 1
fi

case "$MODE" in
    mc|data|full) ;;
    *) echo "ERROR: invalid --mode '$MODE'"; echo; usage; exit 1 ;;
esac

RUN_MC=false
RUN_DATA=false
case "$MODE" in
    mc)   RUN_MC=true ;;
    data) RUN_DATA=true ;;
    full) RUN_MC=true; RUN_DATA=true ;;
esac

# Unique run ID based on timestamp — reruns never collide
RUN_ID="bbww_ft_$(date '+%Y%m%d_%H%M%S')"

# Per-chunk outputs (.coffea + .json sidecars) live alongside the merged
# output, namespaced by RUN_ID so reruns don't overwrite each other.
FINAL_DIR="${OUTPUT_BASE}"
TMP_DIR="${FINAL_DIR}/${RUN_ID}"
mkdir -p "$FINAL_DIR" "$TMP_DIR"

MERGED_JSON="${FINAL_DIR}/${OUTPUT_NAME}.json"

# Shared runner.py flags (no -d / -y / -o; set per chunk below)
COMMON="python runner.py \
    -p bbreww/analysis/processors/hh_bbww_processor.py \
    -m bbreww/metadata/skims_v5 \
    -c bbreww/analysis/metadata/HHbbWW_friendtree.yml \
    --triggers bbreww/metadata/triggers_bbWW.yml \
    --luminosities bbreww/metadata/luminosities_bbWW.yml \
    --friends none \
    -op ${TMP_DIR}/ \
    --condor"

# Temp dir for rerun scripts — cleaned up after merge
RERUN_DIR="$(mktemp -d /tmp/bbww_ft_rerun_XXXXXX)"

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

# Build the active chunk lists based on --mode.
declare -a CHUNK_NAMES=()
declare -a CHUNK_RERUNS=()
declare -a CHUNK_JSONS=()

if $RUN_MC; then
    R1_preEE=$(write_rerun "chunk1_2022_preEE"   "${COMMON} -d ${CHUNK1_DS} -y 2022_preEE   -o chunk1_2022_preEE.coffea")
    R1_EE=$(write_rerun    "chunk1_2022_EE"      "${COMMON} -d ${CHUNK1_DS} -y 2022_EE       -o chunk1_2022_EE.coffea")
    R1_pre=$(write_rerun   "chunk1_2023_preBPix" "${COMMON} -d ${CHUNK1_DS} -y 2023_preBPix -o chunk1_2023_preBPix.coffea")
    R1_B=$(write_rerun     "chunk1_2023_BPix"    "${COMMON} -d ${CHUNK1_DS} -y 2023_BPix     -o chunk1_2023_BPix.coffea")
    R2=$(write_rerun       "chunk2" "${COMMON} -d ${CHUNK2_DS} -y ${ALL_ERAS} -o chunk2.coffea")
    R3=$(write_rerun       "chunk3" "${COMMON} -d ${CHUNK3_DS} -y ${ALL_ERAS} -o chunk3.coffea")
    R4=$(write_rerun       "chunk4" "${COMMON} -d ${CHUNK4_DS} -y ${ALL_ERAS} -o chunk4.coffea")
    R5=$(write_rerun       "chunk5" "${COMMON} -d ${CHUNK5_DS} -y ${ALL_ERAS} -o chunk5.coffea")

    CHUNK_NAMES+=("chunk1_2022_preEE" "chunk1_2022_EE" "chunk1_2023_preBPix" "chunk1_2023_BPix" "chunk2" "chunk3" "chunk4" "chunk5")
    CHUNK_RERUNS+=("$R1_preEE" "$R1_EE" "$R1_pre" "$R1_B" "$R2" "$R3" "$R4" "$R5")
    CHUNK_JSONS+=(
        "${TMP_DIR}/chunk1_2022_preEE.json"
        "${TMP_DIR}/chunk1_2022_EE.json"
        "${TMP_DIR}/chunk1_2023_preBPix.json"
        "${TMP_DIR}/chunk1_2023_BPix.json"
        "${TMP_DIR}/chunk2.json"
        "${TMP_DIR}/chunk3.json"
        "${TMP_DIR}/chunk4.json"
        "${TMP_DIR}/chunk5.json"
    )
fi

if $RUN_DATA; then
    RD_EG=$(write_rerun "data_egamma"     "${COMMON} -d ${DATA_EG_DS} -y ${ALL_ERAS} -o data_egamma.coffea")
    RD_MU=$(write_rerun "data_singlemuon" "${COMMON} -d ${DATA_MU_DS} -y ${ALL_ERAS} -o data_singlemuon.coffea")
    CHUNK_NAMES+=("data_egamma" "data_singlemuon")
    CHUNK_RERUNS+=("$RD_EG" "$RD_MU")
    CHUNK_JSONS+=(
        "${TMP_DIR}/data_egamma.json"
        "${TMP_DIR}/data_singlemuon.json"
    )
fi

# ---------------------------------------------------------------------------
# Helper: open a screen window that runs the chunk's rerun script
# Writes ${RERUN_DIR}/${name}.exit with the exit code so the merge window
# can detect failures and trigger automatic retries.
# ---------------------------------------------------------------------------
window_cmd() {
    local name="$1"
    local rerun="$2"
    local inner="cd ${BARISTA_DIR} && bash ${rerun}; rc=\$?; echo \$rc > ${RERUN_DIR}/${name}.exit; echo; echo '>>> ${name} exited with code '\$rc; exec bash"
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

echo "Run ID:  ${RUN_ID}"
echo "Tmp dir: ${TMP_DIR}"
echo ""

# Create session with the FIRST window (whichever chunk that is for this mode)
FIRST_NAME="${CHUNK_NAMES[0]}"
FIRST_RERUN="${CHUNK_RERUNS[0]}"
echo "Creating screen session '${SESSION}'..."
screen -dmS "$SESSION" -t "$FIRST_NAME" bash -c \
    "cd ${BARISTA_DIR} && bash ${FIRST_RERUN}; rc=\$?; echo \$rc > ${RERUN_DIR}/${FIRST_NAME}.exit; echo; echo '>>> ${FIRST_NAME} exited with code '\$rc; exec bash"

sleep 5

# Spawn remaining windows
for i in $(seq 1 $((${#CHUNK_NAMES[@]} - 1))); do
    window_cmd "${CHUNK_NAMES[$i]}" "${CHUNK_RERUNS[$i]}"
    sleep 5
done

# ---------------------------------------------------------------------------
# Manual merge script — written alongside the chunk outputs so the user can
# run the merge by hand if the auto merge window dies or is killed.
# Uses the same CHUNK_JSONS / MERGED_JSON the auto merge uses.
# ---------------------------------------------------------------------------
MANUAL_MERGE="${TMP_DIR}/manual_merge.sh"
{
    echo "#!/bin/bash"
    echo "# Manual merge for run ${RUN_ID} (mode: ${MODE})"
    echo "# Generated by submit_friendtree_run3.sh at submit time."
    echo "cd \"${BARISTA_DIR}\""
    echo ""
    echo "./run_container python -m src.friendtrees.merge_friend_meta \\"
    echo "    -o \"${MERGED_JSON}\" \\"
    echo "    -i \\"
    n=${#CHUNK_JSONS[@]}
    for i in "${!CHUNK_JSONS[@]}"; do
        if [[ $i -lt $((n - 1)) ]]; then
            echo "        \"${CHUNK_JSONS[$i]}\" \\"
        else
            echo "        \"${CHUNK_JSONS[$i]}\""
        fi
    done
} > "$MANUAL_MERGE"
chmod +x "$MANUAL_MERGE"

# ---------------------------------------------------------------------------
# Merge window — waits 30 min, polls every 5 min for JSON metafiles, merges
# Auto-retries failed chunks (those whose runner exited but JSON is missing).
# ---------------------------------------------------------------------------
MERGE_SCRIPT="$(mktemp /tmp/bbww_ft_merge_XXXXXX.sh)"
cat > "$MERGE_SCRIPT" << MERGEEOF
#!/bin/bash
cd "${BARISTA_DIR}"
log() { echo "[\$(date '+%H:%M:%S')] \$*"; }

CHUNK_NAMES=(
$(printf '    "%s"\n' "${CHUNK_NAMES[@]}")
)
CHUNK_JSONS=(
$(printf '    "%s"\n' "${CHUNK_JSONS[@]}")
)

RERUN_DIR="${RERUN_DIR}"
SESSION="${SESSION}"
BARISTA_DIR="${BARISTA_DIR}"

# Retry budget per chunk (initial attempt counts as #1; up to 3 retries
# beyond that for a total of 4 attempts).
MAX_RETRIES=3
declare -A RETRIES
for name in "\${CHUNK_NAMES[@]}"; do
    RETRIES["\$name"]=0
done

# Re-spawn a chunk window with its rerun script (used to auto-retry on failure).
# Build a self-contained launcher script so screen doesn't have to parse the
# inner command line — sidesteps quoting bugs in 'screen -X screen bash -c ...'.
respawn_chunk() {
    local name="\$1"
    local rerun_script="\${RERUN_DIR}/rerun_\${name}.sh"
    [[ ! -f "\$rerun_script" ]] && { log "ERROR: rerun script missing for \$name"; return 1; }
    rm -f "\${RERUN_DIR}/\${name}.exit"

    local launcher="\${RERUN_DIR}/launch_\${name}_retry\${RETRIES[\$name]}.sh"
    cat > "\$launcher" <<LAUNCHEOF
#!/bin/bash
cd "\${BARISTA_DIR}"
bash "\${rerun_script}"
rc=\\\$?
echo "\\\$rc" > "\${RERUN_DIR}/\${name}.exit"
echo
echo ">>> \${name} (retry) exited with code \\\$rc"
exec bash
LAUNCHEOF
    chmod +x "\$launcher"

    if ! screen -S "\$SESSION" -X screen -t "\${name}_retry\${RETRIES[\$name]}" bash "\$launcher"; then
        log "ERROR: failed to spawn retry window for \$name (screen -X exit non-zero)"
        return 1
    fi
}

log "Run ID:  ${RUN_ID}"
log "Tmp dir: ${TMP_DIR}"
log ""
log "Rerun commands for all chunks:"
for rr in "\${RERUN_DIR}"/rerun_*.sh; do
    log "  bash \$rr"
done

log "Waiting 30 minutes before polling for output files / exit codes..."
sleep 1800

log "Starting to poll for JSON metafiles / exit codes (every 5 minutes)..."
while true; do
    missing=()
    failed_idx=()
    in_flight=0
    for i in "\${!CHUNK_NAMES[@]}"; do
        name="\${CHUNK_NAMES[\$i]}"
        f="\${CHUNK_JSONS[\$i]}"
        if [[ -f "\$f" ]]; then
            continue  # success — JSON metafile exists
        fi
        # JSON missing — check if window has finished (exit-code marker present)
        if [[ -f "\${RERUN_DIR}/\${name}.exit" ]]; then
            rc=\$(cat "\${RERUN_DIR}/\${name}.exit")
            if [[ "\$rc" != "0" ]]; then
                failed_idx+=("\$i")
            else
                # Exit 0 but no JSON — also treat as failure
                failed_idx+=("\$i")
            fi
        else
            missing+=("\$(basename "\$f")")
            in_flight=\$((in_flight + 1))
        fi
    done

    if [[ \${#missing[@]} -eq 0 && \${#failed_idx[@]} -eq 0 ]]; then
        log "All chunks succeeded."
        break
    fi

    # Auto-retry failed chunks.
    for i in "\${failed_idx[@]}"; do
        name="\${CHUNK_NAMES[\$i]}"
        manual_marker="${TMP_DIR}/.\${name}.manual"
        if [[ -f "\$manual_marker" ]]; then
            log "Manual rerun in progress for \$name (marker \$manual_marker); skipping auto-retry."
            log "  (If that manual run died, remove the marker to re-enable auto-retry: rm -f \$manual_marker)"
            in_flight=\$((in_flight + 1))
            continue
        fi
        if [[ "\${RETRIES[\$name]}" -ge "\$MAX_RETRIES" ]]; then
            log "GIVING UP on \$name after \${RETRIES[\$name]} retries"
            log "  To rerun manually (creates a marker so this script won't also retry it):"
            log "    touch \"\$manual_marker\" && cd \${BARISTA_DIR} && ./run_container bash \${RERUN_DIR}/rerun_\${name}.sh; rm -f \"\$manual_marker\""
            continue
        fi
        RETRIES["\$name"]=\$((RETRIES["\$name"] + 1))
        log "Auto-retry \${RETRIES[\$name]}/\$MAX_RETRIES for \$name"
        respawn_chunk "\$name"
        in_flight=\$((in_flight + 1))
    done

    # Bail out if everything failed permanently before polling forever.
    if [[ \$in_flight -eq 0 && \${#failed_idx[@]} -gt 0 ]]; then
        log "All remaining chunks have exhausted retries. Aborting before merge."
        exec bash
    fi

    log "Still waiting on \$in_flight chunk(s): \${missing[*]:-(retried)}"
    sleep 300
done

log "Merging JSON metafiles..."
./run_container python -m src.friendtrees.merge_friend_meta \
    -o "${MERGED_JSON}" \
    -i "\${CHUNK_JSONS[@]}"

log "Cleaning up rerun scripts from \${RERUN_DIR}..."
rm -rf "\${RERUN_DIR}"

log "Done. Merged output: ${MERGED_JSON}"
exec bash
MERGEEOF
chmod +x "$MERGE_SCRIPT"

sleep 5
screen -S "$SESSION" -X screen -t "merge" bash "$MERGE_SCRIPT"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "Mode: ${MODE}"
echo "Screen session '${SESSION}' is running with the following windows:"
for i in "${!CHUNK_NAMES[@]}"; do
    printf "  %d: %s\n" "$i" "${CHUNK_NAMES[$i]}"
done
echo "  ${#CHUNK_NAMES[@]}: merge               — waiting 30 min then polling every 5 min"
echo ""
echo "Tmp dir (per-chunk outputs): ${TMP_DIR}"
echo "Final merged output:         ${MERGED_JSON}"
echo "Manual merge script:         bash ${MANUAL_MERGE}"
echo "  (use if auto-merge dies; verify all JSONs exist first: ls ${TMP_DIR}/*.json)"
echo ""
echo "To rerun a failed chunk: go to its window and hit up-arrow"
echo ""
echo "Attach:          screen -r ${SESSION}"
echo "Switch windows:  Ctrl-A N / Ctrl-A P   or   Ctrl-A \" (list)"
echo "Detach:          Ctrl-A D"
