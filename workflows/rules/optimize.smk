"""
rules/optimize.smk — pass-1 quantile rebinning for the bbWW analysis.

NOTE: this is the quantile-regression *rebinning* step — NOT the norm scan.
The norm is fixed (whatever train.yml says); these rules derive the SR bin edges
from the pass-1 signal/background phh distributions and write them into
hist_templates.py so pass 2 histograms with the optimized binning.

Rules:
  modify_config_pass1  — derive a pass-1 config with dump_signal_phh: true
  cleanup_quantiles    — wipe the EOS quantiles dir (template; instantiated twice)
  quantile_regression  — pass-1 coffea's EOS quantiles -> bin_edges_<region>.txt
  rebin_hist_templates — patch hist_templates.py with the new edges (+ snapshot)

The bin-edge patching logic is the validated implementation from
Snakefile_full_pipeline, kept verbatim so behaviour matches what you've already
tested.
"""

import os

OUT_BASE     = config["output_base"]
QUANTILE_DIR = f"{OUT_BASE}/quantile_fits"
HIST_TEMPLATES = config["hist_templates"]
HIST_SNAPSHOT  = f"{OUT_BASE}/hist_templates_after_rebin.py"
EOS_QUANTILES  = config["eos_quantiles"]

# region key -> (edges file, hist_templates variable name)
REGION_EDGES = {
    "nominal_4j2b": (f"{QUANTILE_DIR}/bin_edges_nominal_4j2b.txt", "phh_variable_nominal"),
    "lowpt_4j2b":   (f"{QUANTILE_DIR}/bin_edges_lowpt_4j2b.txt",   "phh_variable_lowpt"),
    "incl_3j2b":    (f"{QUANTILE_DIR}/bin_edges_incl_3j2b.txt",    "phh_variable_3j2b"),
}


# ── derive a pass-1 config with dump_signal_phh: true ─────────────────────────
# The committed HHbbWW.yml has dump_signal_phh: false. Pass 1 must dump signal
# phh scores to the EOS quantiles dir so quantile_regression has something to
# read. We write a derived config rather than mutating the committed one.
rule modify_config_pass1:
    input:
        config["analysis_config"]
    output:
        f"{OUT_BASE}/HHbbWW_pass1.yml"
    shell:
        r"""
        mkdir -p $(dirname {output})
        sed -e 's|dump_signal_phh:.*|dump_signal_phh: true|' {input} > {output}
        """


# ── quantile regression — pass-1 EOS quantiles -> bin_edges_<region>.txt ──────
# Depends on the pass-1 merged coffea (set as input in the top-level Snakefile)
# so the DAG forces pass 1 to finish first; the regression itself reads the
# quantiles the processor dumped to EOS.
rule quantile_regression:
    input:
        coffea = "PLACEHOLDER_pass1_merged.coffea"
    output:
        nominal  = REGION_EDGES["nominal_4j2b"][0],
        lowpt    = REGION_EDGES["lowpt_4j2b"][0],
        incl3j2b = REGION_EDGES["incl_3j2b"][0],
    log:
        f"{OUT_BASE}/logs/quantile_regression.log"
    params:
        container_wrapper = config["container_wrapper"],
        quantile_dir      = QUANTILE_DIR,
        eos_quantiles     = EOS_QUANTILES,
    shell:
        r"""
        mkdir -p $(dirname {log}) {params.quantile_dir}
        # Force-regenerate edge files so a partial previous run can't leave stale
        # outputs that look "fresh enough" to Snakemake.
        rm -f {output.nominal} {output.lowpt} {output.incl3j2b}

        echo "[$(date)] Running quantile regression on EOS quantiles" > {log}
        {params.container_wrapper} "PYTHONPATH=/srv python -m src.math_tools.quantile_regression \
            --input-dir {params.eos_quantiles} \
            -o {params.quantile_dir}" 2>&1 | tee -a {log}

        for f in {output.nominal} {output.lowpt} {output.incl3j2b}; do
            [ -s "$f" ] || {{ echo "ERROR: $f was not produced" | tee -a {log}; exit 1; }}
        done
        """


# ── rebin hist_templates.py — output is a snapshot copy ───────────────────────
# Snapshotting hist_templates.py as the rule's real output lets Snakemake track
# whether the source file reflects the current edges.
rule rebin_hist_templates:
    input:
        nominal  = REGION_EDGES["nominal_4j2b"][0],
        lowpt    = REGION_EDGES["lowpt_4j2b"][0],
        incl3j2b = REGION_EDGES["incl_3j2b"][0],
    output:
        snapshot = HIST_SNAPSHOT
    log:
        f"{OUT_BASE}/logs/rebin.log"
    params:
        hist_templates = HIST_TEMPLATES,
    shell:
        r"""
        mkdir -p $(dirname {log})
        echo "[$(date)] Updating hist_templates.py with new bin edges" > {log}

        cp {params.hist_templates} {params.hist_templates}.bak

        for pair in "phh_variable_nominal:{input.nominal}" \
                    "phh_variable_lowpt:{input.lowpt}" \
                    "phh_variable_3j2b:{input.incl3j2b}"; do
            var="${{pair%%:*}}"
            edges_file="${{pair#*:}}"
            python3 - "$var" "$edges_file" "{params.hist_templates}" >> {log} 2>&1 <<'PYEOF'
import re, sys, ast
var, edges_file, hist_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(edges_file) as f:
    nums = []
    for line in f:
        s = line.strip()
        if not s or s.startswith('#'): continue
        nums.extend(p.strip() for p in s.split(',') if p.strip())
floats = [float(x) for x in nums]
assert floats[0] == 0.0 and floats[-1] == 1.0
assert all(b > a for a, b in zip(floats, floats[1:]))
new_list = '[' + ', '.join('%.6f' % x for x in floats) + ']'
with open(hist_path) as f: src = f.read()
pat_edges = re.compile(r'(' + re.escape(var) + r'\s*=\s*H\(\s*\(\s*)\[[^\]]*\]', re.MULTILINE)
pat_uniform = re.compile(r'(' + re.escape(var) + r'\s*=\s*H\(\s*\(\s*)[0-9]+\s*,\s*[0-9.eE+-]+\s*,\s*[0-9.eE+-]+', re.MULTILINE)
new_src, n = pat_edges.subn(lambda m: m.group(1) + new_list, src, count=1)
if n == 0:
    new_src, n = pat_uniform.subn(lambda m: m.group(1) + new_list, src, count=1)
if n != 1:
    print('ERROR: pattern matched ' + str(n) + ' times for ' + var, file=sys.stderr); sys.exit(1)
ast.parse(new_src)
with open(hist_path, 'w') as f: f.write(new_src)
print('Updated ' + var + ': ' + str(len(floats)) + ' edges')
PYEOF
        done

        cp {params.hist_templates} {output.snapshot}
        echo "[$(date)] hist_templates.py update complete; snapshot at {output.snapshot}" >> {log}
        """
