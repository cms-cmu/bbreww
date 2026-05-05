#!/bin/bash
# scan_norms.sh
#
# Scans (norm_tt, norm_minor) combinations through the full pipeline:
# train+analyze classifier -> evaluate -> processor -> quantile regression -> rebin -> processor again -> combine.
# Captures combine output per combo. Run once with tmux/nohup; check summary.tsv when done.
#
# Optimization: combo N+1's train+analyze runs in parallel with combo N's processor passes.
# It can't start earlier (writes to ${MODEL}/SvB/ which combo N's evaluate must read first)
# and combo N+1's evaluate must wait until combo N's processor is done (reads EOS friend trees).
#
# Usage:
#   tmux new -s scan
#   bash bbreww/scripts/norm_scan/scan_norms.sh 2>&1 | tee scan_norms.log
#
# Stop with: touch /tmp/scan_norms.stop  (clean break between combos)

set -uo pipefail

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BARISTA_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$BARISTA_DIR"

TRAIN_YML="bbreww/classifier/config/workflows/bbWW_lowpt/svb/train.yml"
HIST_TEMPLATES="bbreww/analysis/helpers/hist_templates.py"
OUT_BASE="output/norm_scan"
SUMMARY_TSV="${OUT_BASE}/summary.tsv"
EOS_QUANTILES="root://cmseos.fnal.gov//store/user/akhanal/HHbbWW/quantiles"
EOS_RESULT_JSON="root://cmseos.fnal.gov//store/user/akhanal/HHbbWW_classifier_lowpt/friend/SvB/result.json"
LOCAL_RESULT_JSON_DIR="bbreww/metadata/classifier/output/"

RUN_TRAIN_ANALYZE="bbreww/scripts/norm_scan/run_train_analyze.sh"
RUN_EVALUATE="bbreww/scripts/norm_scan/run_evaluate.sh"

# State files used to pass prefetched job ids between combos.
PREFETCH_FILE="${OUT_BASE}/.prefetch_train_analyze_jid"
PREFETCH_EVAL_FILE="${OUT_BASE}/.prefetch_evaluate_jid"

mkdir -p "$OUT_BASE"

# ---------------------------------------------------------------------------
# Combinations: powers of 2 up to 16, norm_minor <= norm_tt
# ---------------------------------------------------------------------------
COMBOS=(
    "1 1"
    "2 1" "2 2"
    "4 1" "4 2" "4 4"
    "8 1" "8 2" "8 4" "8 8"
    "16 1" "16 2" "16 4" "16 8" "16 16"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${OUT_BASE}/scan.log"; }

stop_requested() { [[ -f /tmp/scan_norms.stop ]]; }

# Wait until the named jobs finish (any state), polling squeue.
wait_for_jobs() {
    local jids="$1"  # space-separated
    local label="$2"
    for jid in $jids; do
        if ! [[ "$jid" =~ ^[0-9]+$ ]]; then
            log "ERROR: invalid job id '${jid}' for ${label} — refusing to wait"
            return 1
        fi
    done
    log "Waiting for ${label} (jobs: ${jids})..."
    while true; do
        local still_running=0
        for jid in $jids; do
            if squeue -j "$jid" -h 2>/dev/null | grep -q "$jid"; then
                still_running=1
                break
            fi
        done
        [[ $still_running -eq 0 ]] && break
        sleep 60
    done
    log "${label} done."
}

# Edit train.yml in place to set --norm for ttbar (line 19) and minor bkgs (line 28).
set_norms() {
    local n_tt="$1"
    local n_minor="$2"
    sed -i "19s/--norm [0-9]\+/--norm ${n_tt}/" "$TRAIN_YML"
    sed -i "28s/--norm [0-9]\+/--norm ${n_minor}/" "$TRAIN_YML"
    log "train.yml updated: norm_tt=${n_tt}, norm_minor=${n_minor}"
    sed -n '19p;28p' "$TRAIN_YML" | sed 's/^/  /' | tee -a "${OUT_BASE}/scan.log"
}

# Cleanup EOS quantiles dir so quantile regression doesn't double-count.
cleanup_quantiles() {
    log "Cleaning up EOS quantiles dir: ${EOS_QUANTILES}"
    xrdfs root://cmseos.fnal.gov ls /store/user/akhanal/HHbbWW/quantiles 2>/dev/null | \
        while read -r f; do
            xrdfs root://cmseos.fnal.gov rm "$f" 2>/dev/null || true
        done
}

# Replace the bin-edges list literal on a named template variable in hist_templates.py.
# Args: $1 = variable name, $2 = path to edges file.
update_hist_template() {
    local var_name="$1"
    local edges_file="$2"

    [[ ! -f "$edges_file" ]] && { log "ERROR: bin edges file missing: $edges_file"; return 1; }

    python3 - "$var_name" "$edges_file" "$HIST_TEMPLATES" <<'PYEOF' || return 1
import re, sys
var_name, edges_file, hist_path = sys.argv[1], sys.argv[2], sys.argv[3]

with open(edges_file) as f:
    nums = []
    for line in f:
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        nums.extend(p.strip() for p in s.split(',') if p.strip())
if not nums:
    print(f"ERROR: no numeric edges parsed from {edges_file}", file=sys.stderr)
    sys.exit(1)
try:
    floats = [float(x) for x in nums]
except ValueError as e:
    print(f"ERROR: non-numeric edge in {edges_file}: {e}", file=sys.stderr)
    sys.exit(1)
if floats[0] != 0.0 or floats[-1] != 1.0 or any(b <= a for a, b in zip(floats, floats[1:])):
    print(f"ERROR: edges in {edges_file} not strictly increasing on [0,1]", file=sys.stderr)
    sys.exit(1)

new_list = '[' + ', '.join(f'{x:.6f}' for x in floats) + ']'

with open(hist_path) as f:
    src = f.read()

pat = re.compile(
    r'(' + re.escape(var_name) + r'\s*=\s*H\(\s*\(\s*)\[[^\]]*\]',
    re.MULTILINE,
)
new_src, n = pat.subn(lambda m: m.group(1) + new_list, src, count=1)
if n != 1:
    print(f"ERROR: pattern matched {n} times for {var_name} in {hist_path}", file=sys.stderr)
    sys.exit(1)

import ast
try:
    ast.parse(new_src)
except SyntaxError as e:
    print(f"ERROR: edit would produce invalid Python: {e}", file=sys.stderr)
    sys.exit(1)

with open(hist_path, 'w') as f:
    f.write(new_src)
print(f"Updated {var_name}: {len(floats)} edges")
PYEOF
    log "Updated ${HIST_TEMPLATES} variable ${var_name} with edges from ${edges_file}"
}

# Submit submit_all_run3_falcon.sh and return the merge job id.
submit_chunks_and_get_merge_jid() {
    local out
    out=$(bash bbreww/scripts/submit_all_run3_falcon.sh 2>&1 | tee -a "${OUT_BASE}/scan.log")
    echo "$out" | grep -oE "Submitted merge.*— job [0-9]+" | grep -oE "[0-9]+$"
}

# Submit a classifier sub-script via run_container (returns the slurm job id).
submit_classifier_step() {
    local script_path="$1"
    local logfile="$2"
    local out
    out=$(./run_container classifier source "$script_path" 2>&1 | tee "$logfile")
    echo "$out" | awk '/^Submitted batch job/ {print $4}' | tail -1
}

# Submit the next combo's train+analyze in the background (does NOT wait).
# Args: $1 = current combo's norm_tt, $2 = current combo's norm_minor.
# Reads NEXT_COMBO_NORMS env var; uses PREFETCH_FILE to stash the JID for combo N+1.
prefetch_next_train_analyze() {
    local cur_n_tt="$1"
    local cur_n_minor="$2"

    [[ -z "${NEXT_COMBO_NORMS:-}" ]] && return 0
    [[ -f "$PREFETCH_FILE" ]] && return 0  # already prefetched

    local next_n_tt next_n_minor
    read -r next_n_tt next_n_minor <<< "$NEXT_COMBO_NORMS"
    local next_combo_dir="${OUT_BASE}/tt${next_n_tt}_minor${next_n_minor}"
    local next_M="${next_combo_dir}/.done"
    mkdir -p "$next_combo_dir" "$next_M"

    if [[ -f "${next_M}/01_train_analyze" ]]; then
        log "[prefetch] Next combo (${next_n_tt}, ${next_n_minor}) already has train+analyze marker; skipping prefetch"
        return 0
    fi

    log "[prefetch] Setting norms for next combo (${next_n_tt}, ${next_n_minor}) and launching train+analyze in background..."
    set_norms "$next_n_tt" "$next_n_minor"
    local pre_jid
    pre_jid=$(submit_classifier_step "$RUN_TRAIN_ANALYZE" "${next_combo_dir}/train_analyze_submit.log")
    if [[ -n "$pre_jid" ]] && [[ "$pre_jid" =~ ^[0-9]+$ ]]; then
        echo "$pre_jid" > "$PREFETCH_FILE"
        log "[prefetch] Submitted train+analyze for next combo: job ${pre_jid}"
    else
        log "[prefetch] WARNING: could not parse prefetch job id; next combo will train fresh"
    fi
    # IMPORTANT: do NOT restore train.yml to current combo's norms here. The slurm job
    # we just submitted reads train.yml at RUNTIME (when it actually starts on the node),
    # not at submission time. If we revert, the queued job would read the wrong norms.
    # Safe to leave train.yml at next combo's norms because combo N's remaining steps
    # (5-9) don't touch train.yml. Combo N+1's set_norms() at the start will re-confirm.
}

# Wait for the prefetched train+analyze (if any) to finish, then submit evaluate
# for combo N+1 in the background. Saves the evaluate JID so combo N+1's step 2
# can wait on it instead of submitting fresh.
prefetch_next_evaluate() {
    [[ -z "${NEXT_COMBO_NORMS:-}" ]] && return 0
    [[ -f "$PREFETCH_EVAL_FILE" ]] && return 0  # already prefetched
    [[ ! -f "$PREFETCH_FILE" ]] && {
        log "[prefetch-eval] No prefetched train+analyze JID found; skipping evaluate prefetch"
        return 0
    }

    local next_n_tt next_n_minor
    read -r next_n_tt next_n_minor <<< "$NEXT_COMBO_NORMS"
    local next_combo_dir="${OUT_BASE}/tt${next_n_tt}_minor${next_n_minor}"
    local next_M="${next_combo_dir}/.done"
    mkdir -p "$next_combo_dir" "$next_M"

    if [[ -f "${next_M}/02_evaluate" ]]; then
        log "[prefetch-eval] Next combo (${next_n_tt}, ${next_n_minor}) already has evaluate marker; skipping"
        return 0
    fi

    local train_jid
    train_jid=$(cat "$PREFETCH_FILE")
    log "[prefetch-eval] Waiting for prefetched train+analyze (job ${train_jid}) before submitting evaluate..."
    wait_for_jobs "$train_jid" "prefetched train+analyze" || {
        log "[prefetch-eval] WARNING: train+analyze ${train_jid} wait failed; next combo will evaluate fresh"
        return 0
    }

    # Train+analyze is done; mark it for combo N+1 and remove the JID file.
    touch "${next_M}/01_train_analyze"
    rm -f "$PREFETCH_FILE"

    log "[prefetch-eval] Submitting evaluate for next combo (${next_n_tt}, ${next_n_minor})..."
    local eval_jid
    eval_jid=$(submit_classifier_step "$RUN_EVALUATE" "${next_combo_dir}/evaluate_submit.log")
    if [[ -n "$eval_jid" ]] && [[ "$eval_jid" =~ ^[0-9]+$ ]]; then
        echo "$eval_jid" > "$PREFETCH_EVAL_FILE"
        log "[prefetch-eval] Submitted evaluate for next combo: job ${eval_jid}"
    else
        log "[prefetch-eval] WARNING: could not parse evaluate job id; next combo will evaluate fresh"
    fi
}

# ---------------------------------------------------------------------------
# Per-combo pipeline (8 numbered steps)
# ---------------------------------------------------------------------------
# Steps:
#   01_train_analyze        — train classifier + plot ROC/AUC (writes ${MODEL}/SvB/)
#   02_evaluate             — inference using trained model (writes ${SvB}/ friend trees)
#   03_xrdcp_result         — copy result.json from EOS to local
#   04_cleanup_pre          — wipe EOS quantiles dir before processor
#   05_processor_pass1      — first processor run, populates EOS quantiles dir
#   06_quantile_regression  — compute new bin edges from quantiles
#   07_rebin                — write new edges into hist_templates.py
#   08_processor_pass2      — second processor run with new bins
#   09_combine              — combine snakemake workflow → limits
# ---------------------------------------------------------------------------
run_combo() {
    local n_tt="$1"
    local n_minor="$2"
    local combo_dir="${OUT_BASE}/tt${n_tt}_minor${n_minor}"
    mkdir -p "$combo_dir"
    local M="${combo_dir}/.done"
    mkdir -p "$M"

    log "=========================================================="
    log "Starting combo: norm_tt=${n_tt}, norm_minor=${n_minor}"
    log "Output: ${combo_dir}"
    log "=========================================================="

    set_norms "$n_tt" "$n_minor"

    # 1. Train + analyze (may have been prefetched at end of previous combo)
    if [[ -f "${M}/01_train_analyze" ]]; then
        log "[1/9] Skipping train+analyze (already done for this combo)"
    else
        local cls_jid=""
        # Was a prefetch JID staged for us?
        if [[ -f "$PREFETCH_FILE" ]]; then
            cls_jid=$(cat "$PREFETCH_FILE")
            rm -f "$PREFETCH_FILE"
            if [[ "$cls_jid" =~ ^[0-9]+$ ]]; then
                log "[1/9] Found prefetched train+analyze job ${cls_jid}; will wait on it instead of resubmitting"
            else
                log "[1/9] Prefetch file had garbage; submitting fresh train+analyze"
                cls_jid=""
            fi
        fi
        if [[ -z "$cls_jid" ]]; then
            log "[1/9] Submitting train+analyze..."
            cls_jid=$(submit_classifier_step "$RUN_TRAIN_ANALYZE" "${combo_dir}/train_analyze_submit.log")
            if [[ -z "$cls_jid" ]]; then
                log "ERROR: could not parse train+analyze job id; aborting combo"
                return 1
            fi
        fi
        wait_for_jobs "$cls_jid" "train+analyze" || return 1
        touch "${M}/01_train_analyze"
    fi

    # 2. Evaluate (writes EOS friend trees that the processor consumes)
    if [[ -f "${M}/02_evaluate" ]]; then
        log "[2/9] Skipping evaluate (already done)"
    else
        local ev_jid=""
        # Was an evaluate JID prefetched for this combo at the previous combo's step 8b?
        if [[ -f "$PREFETCH_EVAL_FILE" ]]; then
            ev_jid=$(cat "$PREFETCH_EVAL_FILE")
            rm -f "$PREFETCH_EVAL_FILE"
            if [[ "$ev_jid" =~ ^[0-9]+$ ]]; then
                log "[2/9] Found prefetched evaluate job ${ev_jid}; will wait on it instead of resubmitting"
            else
                log "[2/9] Prefetch eval file had garbage; submitting fresh evaluate"
                ev_jid=""
            fi
        fi
        if [[ -z "$ev_jid" ]]; then
            log "[2/9] Submitting evaluate..."
            ev_jid=$(submit_classifier_step "$RUN_EVALUATE" "${combo_dir}/evaluate_submit.log")
            if [[ -z "$ev_jid" ]]; then
                log "ERROR: could not parse evaluate job id; aborting combo"
                return 1
            fi
        fi
        wait_for_jobs "$ev_jid" "evaluate" || return 1
        touch "${M}/02_evaluate"
    fi

    # 3. Copy result.json from EOS
    if [[ -f "${M}/03_xrdcp_result" ]]; then
        log "[3/9] Skipping xrdcp result.json (already done)"
    else
        log "[3/9] Copying result.json from EOS..."
        xrdcp -f "$EOS_RESULT_JSON" "$LOCAL_RESULT_JSON_DIR" 2>&1 | tee "${combo_dir}/xrdcp.log" || return 1
        cp "${LOCAL_RESULT_JSON_DIR}/result.json" "${combo_dir}/result.json"
        touch "${M}/03_xrdcp_result"
    fi

    # 3b. PREFETCH next combo's train+analyze (does NOT block).
    # Safe here: combo N's evaluate is done, so ${MODEL}/SvB/ is no longer needed by combo N.
    # The processor reads only local result.json + EOS friend trees at ${SvB}/,
    # neither of which train+analyze touches.
    prefetch_next_train_analyze "$n_tt" "$n_minor"

    # 4. Pre-pass quantiles cleanup
    if [[ -f "${M}/04_cleanup_pre" ]]; then
        log "[4/9] Skipping pre-pass quantiles cleanup (already done)"
    else
        log "[4/9] Cleaning EOS quantiles dir..."
        cleanup_quantiles
        touch "${M}/04_cleanup_pre"
    fi

    # 5. First processor pass
    if [[ -f "${M}/05_processor_pass1" ]]; then
        log "[5/9] Skipping processor 1st pass (already done)"
    else
        log "[5/9] Submitting processor (1st pass)..."
        local merge_jid
        merge_jid=$(submit_chunks_and_get_merge_jid)
        if [[ -z "$merge_jid" ]]; then
            log "ERROR: could not parse merge job id from submit_all_run3_falcon.sh"
            return 1
        fi
        wait_for_jobs "$merge_jid" "processor 1st pass + merge" || return 1
        touch "${M}/05_processor_pass1"
    fi

    # 6. Quantile regression
    if [[ -f "${M}/06_quantile_regression" ]]; then
        log "[6/9] Skipping quantile regression (already done)"
    else
        log "[6/9] Running quantile regression..."
        ./run_container python -m src.math_tools.quantile_regression \
            --input-dir "$EOS_QUANTILES" \
            -o output/full_run/quantile_fits \
            --n-bins-max 50 2>&1 | tee "${combo_dir}/quantile_regression.log" || return 1

        cp output/full_run/quantile_fits/bin_edges_nominal_4j2b.txt "${combo_dir}/" 2>/dev/null || true
        cp output/full_run/quantile_fits/bin_edges_lowpt_4j2b.txt "${combo_dir}/" 2>/dev/null || true
        touch "${M}/06_quantile_regression"
    fi

    # 7. Rebin: copy bin edges into hist_templates.py
    if [[ -f "${M}/07_rebin" ]]; then
        log "[7/9] Skipping hist_templates.py rebin (already done)"
    else
        log "[7/9] Updating hist_templates.py with new bin edges..."
        cp "$HIST_TEMPLATES" "${combo_dir}/hist_templates.py.bak"
        update_hist_template phh_variable_nominal output/full_run/quantile_fits/bin_edges_nominal_4j2b.txt || return 1
        update_hist_template phh_variable_lowpt   output/full_run/quantile_fits/bin_edges_lowpt_4j2b.txt || return 1
        cp "$HIST_TEMPLATES" "${combo_dir}/hist_templates.py.new"
        touch "${M}/07_rebin"
    fi

    # 8. Second processor pass with new bins
    if [[ -f "${M}/08_processor_pass2" ]]; then
        log "[8/9] Skipping processor 2nd pass (already done)"
    else
        log "[8/9] Submitting processor (2nd pass with new bins)..."
        cleanup_quantiles
        merge_jid=$(submit_chunks_and_get_merge_jid)
        if [[ -z "$merge_jid" ]]; then
            log "ERROR: could not parse merge job id from 2nd submit_all_run3_falcon.sh"
            return 1
        fi
        wait_for_jobs "$merge_jid" "processor 2nd pass + merge" || return 1

        cp output/full_run/output_merged.coffea "${combo_dir}/output_merged.coffea" 2>/dev/null || \
            log "WARNING: output_merged.coffea missing after 2nd pass"
        touch "${M}/08_processor_pass2"
    fi

    # 8b. PREFETCH next combo's evaluate (does NOT block).
    # Safe here: combo N's processor pass 2 is done reading ${SvB}/, so combo N+1's evaluate
    # can overwrite the friend trees there. Combine (step 9) reads only output_merged.coffea,
    # not ${SvB}/, so it's unaffected.
    prefetch_next_evaluate

    # 9. Combine via snakemake
    if [[ -f "${M}/09_combine" ]]; then
        log "[9/9] Skipping combine (already done)"
    else
        log "[9/9] Running combine snakemake workflow..."
        if [[ -f "${combo_dir}/output_merged.coffea" ]]; then
            cp "${combo_dir}/output_merged.coffea" output/full_run/output_merged.coffea
        else
            log "WARNING: per-combo output_merged.coffea missing for ${combo_dir}; combine may use stale data"
        fi

        ./run_container snakemake \
            --snakefile bbreww/workflows/Snakefile_combine \
            --configfile bbreww/workflows/config_combine.yml \
            --cores 1 \
            --jobs 1 \
            --latency-wait 60 \
            --forceall 2>&1 | tee "${combo_dir}/combine.log"

        if [[ -f output/combine/summary.txt ]]; then
            cp output/combine/summary.txt "${combo_dir}/limits_summary.txt"
        else
            log "WARNING: output/combine/summary.txt not produced by snakemake"
        fi

        touch "${M}/09_combine"
    fi

    # Parse the combined Expected 50.0% line from the per-combo limits_summary.txt
    local r50="parse_failed"
    if [[ -f "${combo_dir}/limits_summary.txt" ]]; then
        r50=$(awk '/^--- combined ---/{f=1; next} /^---/{f=0} f && /Expected[[:space:]]+50\.0%/{print; exit}' \
              "${combo_dir}/limits_summary.txt" | awk -F'< ' '{print $2}' | tr -d '\r ')
        [[ -z "$r50" ]] && r50="parse_failed"
    fi

    if [[ ! -f "$SUMMARY_TSV" ]]; then
        echo -e "norm_tt\tnorm_minor\tcombined_r50\ttimestamp" > "$SUMMARY_TSV"
    fi
    echo -e "${n_tt}\t${n_minor}\t${r50}\t$(date '+%Y-%m-%d %H:%M:%S')" >> "$SUMMARY_TSV"

    log "Combo done: norm_tt=${n_tt}, norm_minor=${n_minor}, combined_r50=${r50}"
}

# ---------------------------------------------------------------------------
# Main loop — passes NEXT_COMBO_NORMS env var into run_combo so it knows
# which combo to prefetch a classifier for.
# ---------------------------------------------------------------------------
log "Starting scan with ${#COMBOS[@]} combinations"
log "Stop at any time: touch /tmp/scan_norms.stop"

for i in "${!COMBOS[@]}"; do
    if stop_requested; then
        log "Stop requested. Exiting."
        rm -f /tmp/scan_norms.stop
        break
    fi

    read -r n_tt n_minor <<< "${COMBOS[$i]}"
    next_i=$((i + 1))
    if [[ $next_i -lt ${#COMBOS[@]} ]]; then
        export NEXT_COMBO_NORMS="${COMBOS[$next_i]}"
    else
        export NEXT_COMBO_NORMS=""
    fi

    run_combo "$n_tt" "$n_minor" || log "Combo (${n_tt}, ${n_minor}) FAILED, continuing"
done

log "Scan finished. Summary at ${SUMMARY_TSV}"
