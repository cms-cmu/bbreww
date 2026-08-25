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
# Modes (selected via --mode):
#   mc   — MC samples only:
#            chunk1_<era>  (GluGlu + TTToSemiLeptonic, one per era)
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
#   bash bbreww/scripts/submit_all_run3.sh --mode <mc|data|full> [--output-base DIR]
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

# Anchor every relative path below to barista root, regardless of where the
# user invokes this script from.
cd "$BARISTA_DIR"

OUTPUT_BASE="output"
OUTPUT_NAME="output_merged.coffea"
SESSION="bbww_run3"
MODE=""

usage() {
    cat <<EOF
Usage: $0 --mode <mc|data|full> [--output-base DIR] [-op NAME]

Modes:
  mc    Run MC chunks only (chunk1_<era>, chunk2, chunk3, chunk4, chunk5)
  data  Run the data chunks only (data_egamma + data_singlemuon, all eras)
  full  Run MC and Data chunks together

Optional:
  --output-base DIR    Where to write outputs (default: output)
  -op NAME             Filename for the final merged coffea
                       (default: output_merged.coffea). The file is written
                       to <output-base>/full_run/<NAME>.
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-base) OUTPUT_BASE="$2"; shift 2 ;;
        -op)           OUTPUT_NAME="$2"; shift 2 ;;
        --mode)        MODE="$2"; shift 2 ;;
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

OUTPUT_DIR="${OUTPUT_BASE}/full_run"
mkdir -p "$OUTPUT_DIR"

# Remove pre-existing chunk outputs so a fresh run doesn't pick up stale
# histograms with different binning. (output_merged.coffea is left alone.)
echo "Cleaning pre-existing chunk outputs in ${OUTPUT_DIR}..."
if $RUN_MC; then
    rm -f "${OUTPUT_DIR}"/output_chunk1_2022_preEE.coffea \
          "${OUTPUT_DIR}"/output_chunk1_2022_EE.coffea \
          "${OUTPUT_DIR}"/output_chunk1_2023_preBPix.coffea \
          "${OUTPUT_DIR}"/output_chunk1_2023_BPix.coffea \
          "${OUTPUT_DIR}"/output_chunk2.coffea \
          "${OUTPUT_DIR}"/output_chunk3.coffea \
          "${OUTPUT_DIR}"/output_chunk4.coffea \
          "${OUTPUT_DIR}"/output_chunk5.coffea
fi
if $RUN_DATA; then
    rm -f "${OUTPUT_DIR}"/output_data_egamma.coffea \
          "${OUTPUT_DIR}"/output_data_singlemuon.coffea
fi

# Wipe any stale manual-rerun markers (.<chunk>.manual) left by a previous run,
# so a fresh submission's auto-retry isn't blocked by a leftover marker.
rm -f "${OUTPUT_DIR}"/.*.manual 2>/dev/null || true

# Wipe EOS quantiles dir IF dump_signal_phh is enabled in HHbbWW.yml.
# This prevents quantile regression from picking up pkls written by an
# earlier run. Path matches the one hardcoded in
# bbreww/analysis/processors/hh_bbww_processor.py:390.
if grep -qE '^\s*dump_signal_phh:\s*true' bbreww/analysis/metadata/HHbbWW.yml; then
    echo "dump_signal_phh=true — wiping EOS quantiles dir..."
    BEFORE=$(xrdfs root://cmseos.fnal.gov ls /store/user/akhanal/HHbbWW/quantiles 2>/dev/null | wc -l)
    echo "  files before: ${BEFORE}"
    mapfile -t TO_DELETE < <(xrdfs root://cmseos.fnal.gov ls /store/user/akhanal/HHbbWW/quantiles 2>/dev/null || true)
    FAILED=0
    for f in "${TO_DELETE[@]}"; do
        if ! xrdfs root://cmseos.fnal.gov rm "$f" 2>/dev/null; then
            FAILED=$((FAILED + 1))
        fi
    done
    AFTER=$(xrdfs root://cmseos.fnal.gov ls /store/user/akhanal/HHbbWW/quantiles 2>/dev/null | wc -l)
    echo "  attempted: ${#TO_DELETE[@]}  failed: ${FAILED}  files after: ${AFTER}"
    if [[ "$AFTER" -ne 0 ]]; then
        echo "WARNING: ${AFTER} pkl file(s) remain after cleanup (may be from a concurrent processor still running)." >&2
    fi
fi

# Shared runner.py flags (no -d / -y; set per chunk below)
COMMON="python runner.py \
    -p bbreww/analysis/processors/hh_bbww_processor.py \
    -m bbreww/metadata/skims_v5 \
    -c bbreww/analysis/metadata/HHbbWW.yml \
    --triggers bbreww/metadata/triggers_bbWW.yml \
    --luminosities bbreww/metadata/luminosities_bbWW.yml \
    --friends none \
    -op ${OUTPUT_DIR}/ \
    --condor"

ALL_ERAS="2022_preEE 2022_EE 2023_preBPix 2023_BPix"
CHUNK1_DS="GluGluToHHTo2B2VLNu2J_kl_1p00 TTToSemiLeptonic GluGlutoHHto2B2Zto2L2Q_kl_1p00 GluGlutoHHto2B2Tau_kl_1p00"
DATA_EG_DS="data__EGamma"
DATA_MU_DS="data__SingleMuon"

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

# Build the active chunk lists based on --mode.
declare -a CHUNK_NAMES=()
declare -a CHUNK_RERUNS=()
declare -a CHUNK_FILES=()

if $RUN_MC; then
    R1_preEE=$(write_rerun "chunk1_2022_preEE"   "${COMMON} -d ${CHUNK1_DS} -y 2022_preEE    -o output_chunk1_2022_preEE.coffea")
    R1_EE=$(write_rerun    "chunk1_2022_EE"      "${COMMON} -d ${CHUNK1_DS} -y 2022_EE        -o output_chunk1_2022_EE.coffea")
    R1_pre=$(write_rerun   "chunk1_2023_preBPix" "${COMMON} -d ${CHUNK1_DS} -y 2023_preBPix  -o output_chunk1_2023_preBPix.coffea")
    R1_B=$(write_rerun     "chunk1_2023_BPix"    "${COMMON} -d ${CHUNK1_DS} -y 2023_BPix      -o output_chunk1_2023_BPix.coffea")
    R2=$(write_rerun       "chunk2"  "${COMMON} -d TTToHadronic TTTo2L2Nu -y ${ALL_ERAS} -o output_chunk2.coffea")
    R3=$(write_rerun       "chunk3"  "${COMMON} -d WtoLNu-2Jets_0J WtoLNu-2Jets_1J WtoLNu-2Jets_2J -y ${ALL_ERAS} -o output_chunk3.coffea")
    R4=$(write_rerun       "chunk4"  "${COMMON} -d TbarWplustoLNu2Q TbarWplusto2L2Nu TWminustoLNu2Q TWminusto2L2Nu -y ${ALL_ERAS} -o output_chunk4.coffea")
    R5=$(write_rerun       "chunk5"  "${COMMON} -d TBbarQ TbarBQ TBbartoLplusNuBbar TbarBtoLminusNuB -y ${ALL_ERAS} -o output_chunk5.coffea")

    CHUNK_NAMES+=("chunk1_2022_preEE" "chunk1_2022_EE" "chunk1_2023_preBPix" "chunk1_2023_BPix" "chunk2" "chunk3" "chunk4" "chunk5")
    CHUNK_RERUNS+=("$R1_preEE" "$R1_EE" "$R1_pre" "$R1_B" "$R2" "$R3" "$R4" "$R5")
    CHUNK_FILES+=(
        "${OUTPUT_DIR}/output_chunk1_2022_preEE.coffea"
        "${OUTPUT_DIR}/output_chunk1_2022_EE.coffea"
        "${OUTPUT_DIR}/output_chunk1_2023_preBPix.coffea"
        "${OUTPUT_DIR}/output_chunk1_2023_BPix.coffea"
        "${OUTPUT_DIR}/output_chunk2.coffea"
        "${OUTPUT_DIR}/output_chunk3.coffea"
        "${OUTPUT_DIR}/output_chunk4.coffea"
        "${OUTPUT_DIR}/output_chunk5.coffea"
    )
fi

if $RUN_DATA; then
    RD_EG=$(write_rerun "data_egamma"     "${COMMON} -d ${DATA_EG_DS} -y ${ALL_ERAS} -o output_data_egamma.coffea")
    RD_MU=$(write_rerun "data_singlemuon" "${COMMON} -d ${DATA_MU_DS} -y ${ALL_ERAS} -o output_data_singlemuon.coffea")
    CHUNK_NAMES+=("data_egamma" "data_singlemuon")
    CHUNK_RERUNS+=("$RD_EG" "$RD_MU")
    CHUNK_FILES+=(
        "${OUTPUT_DIR}/output_data_egamma.coffea"
        "${OUTPUT_DIR}/output_data_singlemuon.coffea"
    )
fi

# ---------------------------------------------------------------------------
# Helper: open a screen window that runs the chunk's rerun script
# The rerun script is the first thing in shell history — just hit up-arrow
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

# Create session with the FIRST window (whichever chunk that is for this mode)
FIRST_NAME="${CHUNK_NAMES[0]}"
FIRST_RERUN="${CHUNK_RERUNS[0]}"
echo "Creating screen session '${SESSION}'..."
screen -dmS "$SESSION" -t "$FIRST_NAME" bash -c \
    "cd ${BARISTA_DIR} && bash ${FIRST_RERUN}; rc=\$?; echo \$rc > ${RERUN_DIR}/${FIRST_NAME}.exit; echo; echo '>>> ${FIRST_NAME} exited with code '\$rc; exec bash"

sleep 5  # give screen time to initialise

# Spawn remaining windows
for i in $(seq 1 $((${#CHUNK_NAMES[@]} - 1))); do
    window_cmd "${CHUNK_NAMES[$i]}" "${CHUNK_RERUNS[$i]}"
    sleep 5
done

# ---------------------------------------------------------------------------
# Merge window — waits 30 min, polls every 5 min, merges, then cleans up
# ---------------------------------------------------------------------------
MERGE_SCRIPT="$(mktemp /tmp/bbww_merge_XXXXXX.sh)"
cat > "$MERGE_SCRIPT" << MERGEEOF
#!/bin/bash
cd "${BARISTA_DIR}"
log() { echo "[\$(date '+%H:%M:%S')] \$*"; }

# Chunk name -> expected coffea output file (parallel arrays since this is bash 4)
CHUNK_NAMES=(
$(printf '    "%s"\n' "${CHUNK_NAMES[@]}")
)
CHUNK_FILES=(
$(printf '    "%s"\n' "${CHUNK_FILES[@]}")
)

RERUN_DIR="${RERUN_DIR}"
SESSION="${SESSION}"
BARISTA_DIR="${BARISTA_DIR}"

# Retry budget per chunk (initial attempt counts as #1; we allow up to 3 retries
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
    # Wipe exit-code marker so we can detect the new attempt's status
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

# Durable, copy-pasteable merge script covering ALL chunks. Written to the output
# dir so it survives RERUN_DIR cleanup. Lets the user merge after manual reruns.
# CHUNK_FILES is baked in at generation time as a flat space-separated list.
MERGE_ALL_SCRIPT="${OUTPUT_DIR}/merge_all.sh"
write_merge_script() {
    {
        echo "#!/bin/bash"
        echo "# Merge all chunks for this run. Re-run any failed chunk first, then run this."
        echo "cd \"${BARISTA_DIR}\""
        echo "./run_container python src/tools/merge_coffea_files.py \\\\"
        echo "    -o \"${OUTPUT_DIR}/${OUTPUT_NAME}\" \\\\"
        echo "    -f ${CHUNK_FILES[*]}"
    } > "\${MERGE_ALL_SCRIPT}"
    chmod +x "\${MERGE_ALL_SCRIPT}"
}

log "Rerun commands for all chunks:"
for rr in "\${RERUN_DIR}"/rerun_*.sh; do
    log "  bash \$rr"
done

# Retries start after RETRY_DELAY; merging is blocked until MERGE_FLOOR has
# elapsed from launch (even if all chunks finish sooner).
LAUNCH_TS=\$(date +%s)
RETRY_DELAY=900   # 15 min before first poll/retry
MERGE_FLOOR=1800  # 30 min minimum age before merge is allowed

log "Waiting \$((RETRY_DELAY / 60)) minutes before polling/retrying..."
sleep \$RETRY_DELAY

log "Starting to poll for output files / exit codes (every 5 minutes)..."
while true; do
    missing=()
    failed_idx=()
    in_flight=0
    for i in "\${!CHUNK_NAMES[@]}"; do
        name="\${CHUNK_NAMES[\$i]}"
        f="\${CHUNK_FILES[\$i]}"
        if [[ -f "\$f" ]]; then
            continue  # success — output exists
        fi
        # Output missing — check if the window has finished (exit-code marker present)
        if [[ -f "\${RERUN_DIR}/\${name}.exit" ]]; then
            rc=\$(cat "\${RERUN_DIR}/\${name}.exit")
            if [[ "\$rc" != "0" ]]; then
                failed_idx+=("\$i")
            else
                # Exit 0 but no output file — also treat as failure
                failed_idx+=("\$i")
            fi
        else
            missing+=("\$(basename "\$f")")
            in_flight=\$((in_flight + 1))
        fi
    done

    if [[ \${#missing[@]} -eq 0 && \${#failed_idx[@]} -eq 0 ]]; then
        # All chunks done — but don't merge until MERGE_FLOOR has elapsed.
        elapsed=\$(( \$(date +%s) - LAUNCH_TS ))
        if [[ \$elapsed -lt \$MERGE_FLOOR ]]; then
            remain=\$(( MERGE_FLOOR - elapsed ))
            log "All chunks succeeded, but holding merge until 30 min floor (\${remain}s remaining)..."
            sleep \$remain
        fi
        log "All chunks succeeded."
        break
    fi

    # Auto-retry failed chunks (those whose runner exited but output is missing).
    for i in "\${failed_idx[@]}"; do
        name="\${CHUNK_NAMES[\$i]}"
        # If a manual rerun is in progress for this chunk (marker present), the
        # user owns it — treat as in-flight and do NOT auto-retry (avoids two
        # runners racing to write the same output).
        manual_marker="${OUTPUT_DIR}/.\${name}.manual"
        if [[ -f "\$manual_marker" ]]; then
            log "Manual rerun in progress for \$name (marker \$manual_marker); skipping auto-retry."
            log "  (If that manual run died, remove the marker to re-enable auto-retry: rm -f \$manual_marker)"
            in_flight=\$((in_flight + 1))
            continue
        fi
        if [[ "\${RETRIES[\$name]}" -ge "\$MAX_RETRIES" ]]; then
            log "GIVING UP on \$name after \${RETRIES[\$name]} retries"
            log "  To rerun this chunk manually (creates a marker so this script won't also retry it):"
            log "    touch \"\$manual_marker\" && cd \${BARISTA_DIR} && ./run_container bash \${RERUN_DIR}/rerun_\${name}.sh; rm -f \"\$manual_marker\""
            log "  (If \${RERUN_DIR} is cleaned up before you retry, run instead:)"
            rerun_script="\${RERUN_DIR}/rerun_\${name}.sh"
            if [[ -f "\$rerun_script" ]]; then
                # Extract the underlying runner.py invocation for a copy-pasteable fallback.
                cmd=\$(grep -E '^\./run_container ' "\$rerun_script" | head -1)
                log "    touch \"\$manual_marker\" && cd \${BARISTA_DIR} && \$cmd; rm -f \"\$manual_marker\""
            fi
            continue
        fi
        RETRIES["\$name"]=\$((RETRIES["\$name"] + 1))
        log "Auto-retry \${RETRIES[\$name]}/\$MAX_RETRIES for \$name"
        respawn_chunk "\$name"
        in_flight=\$((in_flight + 1))
    done

    # If everything failed permanently, bail out before infinite polling.
    if [[ \$in_flight -eq 0 && \${#failed_idx[@]} -gt 0 ]]; then
        log "All remaining chunks have exhausted retries. Aborting before merge."
        write_merge_script
        log "A standalone merge script was written to: \${MERGE_ALL_SCRIPT}"
        log "After you manually rerun the failed chunk(s), merge everything with:"
        log "    cd \${BARISTA_DIR} && bash \${MERGE_ALL_SCRIPT}"
        log ""
        log "Merge command (for copy-paste):"
        log "    cd \${BARISTA_DIR} && ./run_container python src/tools/merge_coffea_files.py -o \"${OUTPUT_DIR}/${OUTPUT_NAME}\" -f \${CHUNK_FILES[*]}"
        exec bash
    fi

    log "Still waiting on \$in_flight chunk(s): \${missing[*]:-(retried)}"
    sleep 300
done

log "Merging..."
./run_container python src/tools/merge_coffea_files.py \\
    -o "${OUTPUT_DIR}/${OUTPUT_NAME}" \\
    -f "\${CHUNK_FILES[@]}"

log "Cleaning up rerun scripts from \${RERUN_DIR}..."
rm -rf "\${RERUN_DIR}"

log "Done. Merged output: ${OUTPUT_DIR}/${OUTPUT_NAME}"
exec bash
MERGEEOF
chmod +x "$MERGE_SCRIPT"

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
echo "To rerun a failed chunk: go to its window and hit up-arrow"
echo ""
echo "Attach:          screen -r ${SESSION}"
echo "Switch windows:  Ctrl-A N / Ctrl-A P   or   Ctrl-A \" (list)"
echo "Detach:          Ctrl-A D"
echo "Output dir:      ${OUTPUT_DIR}/"
