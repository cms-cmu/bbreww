"""
rules/train.smk — optional ML retraining for the bbWW analysis.

Active only when config["retrain"] is true. When false, the top-level Snakefile
does not depend on these rules and the processor uses whatever friend-tree
manifests HHbbWW.yml already points at.

Rules:
  train     — run the METRegressor training (run_train.sh)
  evaluate  — produce friend trees (run_eval.sh) AND refresh the committed
              result_test.json manifest from EOS, so the processor reads THIS
              run's friend trees rather than the frozen baseline.

The friend trees themselves live on EOS (not local files Snakemake can track),
so each rule's real output is a small marker/log artifact written at completion.
The refreshed manifest (result_test.json) is the meaningful local artifact the
processor consumes; it is declared as evaluate's output so the DAG correctly
re-runs the processor when retraining happens.

These steps submit to SLURM internally via `run_container classifier`
(./src/pyml.py auto-submits using slurm.conf) — they are localrules here so
Snakemake does not also wrap them in its own sbatch job.
"""

import os

OUT_BASE = config["output_base"]
TRAIN_DIR = f"{OUT_BASE}/train"

# Paths from run_eval.sh / scan_norms_regressor.sh — the EOS manifest the
# evaluate step writes, and the committed local manifest the processor reads.
MET_FRIEND_EOS = "root://cmseos.fnal.gov//store/user/akhanal/HHbbWW_MET_regressor_test/friend/met_regressor"
MANIFEST_EOS   = f"{MET_FRIEND_EOS}/result.json"
MANIFEST_LOCAL = "bbreww/metadata/regressor/output/result_test.json"

RUN_TRAIN = "bbreww/classifier/config/workflows/METRegressor/run_train.sh"
RUN_EVAL  = "bbreww/classifier/config/workflows/METRegressor/run_eval.sh"


rule train:
    output:
        marker = f"{TRAIN_DIR}/train.done"
    log:
        f"{OUT_BASE}/logs/train.log"
    params:
        container_wrapper = config["container_wrapper"],
        run_train = RUN_TRAIN,
    shell:
        r"""
        mkdir -p $(dirname {log}) $(dirname {output.marker})
        echo "[$(date)] Submitting MET regressor training" | tee {log}
        {params.container_wrapper} classifier {params.run_train} 2>&1 | tee -a {log}
        echo "[$(date)] training step returned" | tee -a {log}
        date > {output.marker}
        """


rule evaluate:
    input:
        f"{TRAIN_DIR}/train.done"
    output:
        # The committed manifest (MANIFEST_LOCAL) is refreshed as a SIDE EFFECT,
        # not declared as a tracked output — declaring it would make Snakemake
        # skip this rule whenever the committed file already exists, so retrain
        # would never run. The real tracked output is the marker, which lives in
        # the (fresh) output tree and is thus always regenerated on a clean run.
        marker = f"{TRAIN_DIR}/evaluate.done",
    log:
        f"{OUT_BASE}/logs/evaluate.log"
    params:
        container_wrapper = config["container_wrapper"],
        run_eval = RUN_EVAL,
        manifest_eos = MANIFEST_EOS,
        manifest_local = MANIFEST_LOCAL,
    shell:
        r"""
        mkdir -p $(dirname {log}) $(dirname {output.marker})
        echo "[$(date)] Submitting MET regressor evaluation (friend trees)" | tee {log}
        {params.container_wrapper} classifier {params.run_eval} 2>&1 | tee -a {log}

        # Refresh the committed manifest so the processor reads THIS run's friend
        # trees (mirrors scan_norms_regressor.sh step 2a). Without this the
        # processor would read the frozen baseline friend trees.
        echo "[$(date)] Refreshing manifest {params.manifest_eos} -> {params.manifest_local}" | tee -a {log}
        xrdcp -f {params.manifest_eos} {params.manifest_local} 2>&1 | tee -a {log}

        [ -s {params.manifest_local} ] || {{ echo "ERROR: manifest {params.manifest_local} not refreshed" | tee -a {log}; exit 1; }}
        date > {output.marker}
        echo "[$(date)] evaluate complete" | tee -a {log}
        """
