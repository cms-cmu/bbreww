"""
rules/processor.smk — generic processor + merge rules for the bbWW analysis.

These are *template* rules. The top-level Snakefile_analysis imports this as a
module and instantiates them per chunk / per pass via:

    use rule analysis_processor from processor as pass1_<chunk> with:
        output: ...
        params: ...

Each instantiated `analysis_processor` rule runs one `runner.py` call (the same
invocation submit_all_run3_falcon.sh issues per chunk), and `merge_coffea`
merges a pass's per-chunk coffea files into one.

Execution backend is Snakemake's SLURM executor — on falcon, run_container
auto-injects software/snakemake/profiles/falcon (executor: slurm), so each rule
instance becomes its own sbatch job and Snakemake (not a hand-rolled
--dependency chain) owns the DAG and parallelism.

NOTE on resources: a bare integer `runtime` in a *rule* directive is parsed as
MINUTES (so runtime=240 -> 4h). The same bare integer in a *profile* YAML is
parsed as seconds — hence the falcon profile uses "8h". Keep rule-level runtime
in minutes.
"""


# ── one processor job (one chunk) ─────────────────────────────────────────────
# The defaults below are overridden by `use rule ... with:` in the top-level
# Snakefile. `params.config` is supplied per pass (pass 1 uses the
# dump_signal_phh=true variant; pass 2 uses the committed config).
# IMPORTANT: with --slurm, runner.py itself becomes a Dask scheduler that
# submits its OWN fleet of SLURM worker jobs (dask_jobqueue.SLURMCluster,
# scaling 1..max_workers from the runner config). For that to work the way it
# is designed — calling the host sbatch via software/slurm/sbatch — runner.py
# must run on the LOGIN NODE, not inside a Snakemake-submitted compute job
# (which would be sbatch-from-inside-sbatch). So the instantiated pass1/pass2
# processor rules are declared `localrules` in Snakefile_analysis: Snakemake
# runs this shell directly on the login node, and runner.py fans the work out
# across compute nodes. No per-rule SLURM `resources` here — worker resources
# (slurm_cores, worker_memory, max_workers, slurm_qos) come from the runner
# config, not from Snakemake.
rule analysis_processor:
    # Match submit_all_run3.sh: up to 3 retries per chunk (4 total attempts).
    # Snakemake already re-runs when the declared output is missing after the
    # job exits, covering the script's "exit 0 but no coffea = failure" case.
    retries: 3
    output:
        coffea = "PLACEHOLDER.coffea"
    log:
        "PLACEHOLDER.log"
    params:
        datasets         = "",
        year             = "",
        config           = config["analysis_config"],
        processor        = config["processor"],
        metadata         = config["metadata"],
        triggers         = config["triggers"],
        luminosities     = config["luminosities"],
        container_wrapper = config["container_wrapper"],
        submit_flag      = config.get("submit_flag", "--slurm"),
    resources:
        # Each --slurm chunk spins up its OWN Dask SLURMCluster. Running many at
        # once oversubscribes the partition (worker-job storm → KilledWorker), so
        # we gate concurrency on a custom `dask_cluster` resource. The global
        # ceiling is set ONLY in the falcon profile (resources: dask_cluster=1),
        # which serializes processor chunks on falcon. LPC's profile sets no such
        # ceiling, so chunks there stay parallel (LPC uses --condor, not a Dask
        # SLURMCluster, so there is no storm to avoid).
        dask_cluster = 1,
    shell:
        r"""
        mkdir -p $(dirname {log}) $(dirname {output.coffea})
        echo "[$(date)] processor ({params.submit_flag}): datasets='{params.datasets}' year='{params.year}'" | tee {log}
        echo "[$(date)] config={params.config}" | tee -a {log}

        # runner.py fans work out to compute nodes itself: --slurm (falcon, Dask
        # SLURMCluster) or --condor (LPC, Dask LPCCondorCluster). The flag is
        # chosen device-aware in Snakefile_full_pipeline. runner.py writes
        # <-op>/<-o>, so we point -op at the output's directory, -o at its basename.
        {params.container_wrapper} python runner.py \
            -p {params.processor} \
            -m {params.metadata} \
            -c {params.config} \
            --triggers {params.triggers} \
            --luminosities {params.luminosities} \
            --friends none \
            -d {params.datasets} \
            -y {params.year} \
            -op $(dirname {output.coffea})/ \
            -o $(basename {output.coffea}) \
            {params.submit_flag} \
            2>&1 | tee -a {log}

        [ -f {output.coffea} ] || {{ echo "ERROR: {output.coffea} not produced" | tee -a {log}; exit 1; }}
        echo "[$(date)] done -> {output.coffea}" | tee -a {log}
        """


# ── merge a pass's per-chunk coffea files into one ────────────────────────────
# `input` (the list of chunk coffea files) and `output` are set per pass via
# `use rule ... with:`.
rule merge_coffea:
    input:
        ["PLACEHOLDER_chunk.coffea"]
    output:
        merged = "PLACEHOLDER_merged.coffea"
    log:
        "PLACEHOLDER_merge.log"
    params:
        container_wrapper = config["container_wrapper"],
    resources:
        cpus_per_task = 4,
        mem_mb        = 32000,
        runtime       = 60,
        slurm_partition = "work",
        qos           = "cpu_light",
    shell:
        r"""
        mkdir -p $(dirname {log}) $(dirname {output.merged})
        echo "[$(date)] merging into {output.merged}" | tee {log}
        {params.container_wrapper} python src/tools/merge_coffea_files.py \
            -o {output.merged} \
            -f {input} \
            2>&1 | tee -a {log}
        [ -f {output.merged} ] || {{ echo "ERROR: {output.merged} not produced" | tee -a {log}; exit 1; }}
        echo "[$(date)] merged -> {output.merged}" | tee -a {log}
        """
