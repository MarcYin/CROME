nextflow.enable.dsl = 2

def sanitizeToken(value) {
    def raw = (value ?: 'run').toString().trim()
    def cleaned = raw.replaceAll(/[^A-Za-z0-9_-]+/, '-').replaceAll(/^-+|-+$/, '')
    return cleaned ? cleaned : 'run'
}

def shellQuote(value) {
    def text = value.toString()
    return "'${text.replace("'", "'\"'\"'")}'"
}

def boolParam(value, defaultValue = false) {
    if (value == null) {
        return defaultValue
    }
    if (value instanceof Boolean) {
        return value
    }
    return value.toString().toBoolean()
}

def requireParam(name) {
    if (params[name] == null || params[name].toString().trim().isEmpty()) {
        error "Missing required parameter: --${name.replace('_', '-')}"
    }
}

params.python = params.python ?: 'python'
params.run_label = params.run_label ?: 'alphaearth'
params.run_pooled_model = boolParam(params.run_pooled_model, true)
params.no_predict = boolParam(params.no_predict, false)
params.fail_on_empty_labels = boolParam(params.fail_on_empty_labels, false)

requireParam('year')
requireParam('reference_path')
if (!params.manifest_path && !params.feature_input) {
    error 'Provide either --manifest_path or --feature_input.'
}

def runLabel = sanitizeToken(params.run_label)
def tileNJobs = params.tile_n_jobs != null ? params.tile_n_jobs : params.tile_cpus
def pooledNJobs = params.pooled_n_jobs != null ? params.pooled_n_jobs : params.pooled_cpus

process PREPARE_TILE_BATCH {
    label 'control'
    tag "${runLabel}"

    output:
    path 'batch_manifest.json', emit: batch_manifest
    path 'tile_manifest_paths.txt', emit: tile_plan_paths

    script:
    def inputArg = params.manifest_path
        ? "--manifest-path ${shellQuote(params.manifest_path)}"
        : "--feature-input ${shellQuote(params.feature_input)}"
    def command = [
        "${params.python} -m crome.cli prepare-tile-batch",
        inputArg,
        "--reference-path ${shellQuote(params.reference_path)}",
        "--year ${params.year}",
        "--output-root ${shellQuote(params.output_root)}",
        "--aoi-label ${shellQuote(runLabel)}",
        "--label-column ${shellQuote(params.label_column)}",
        "--geometry-column ${shellQuote(params.geometry_column)}",
        "--label-mode ${shellQuote(params.label_mode)}",
        "--overlap-policy ${shellQuote(params.overlap_policy)}",
        "--nodata-label ${params.nodata_label}",
        "--test-size ${params.test_size}",
        "--random-state ${params.random_state}",
        "--n-estimators ${params.n_estimators}",
        "--n-jobs ${tileNJobs}",
    ]
    if (params.max_train_rows != null) {
        command << "--max-train-rows ${params.max_train_rows}"
    }
    if (boolParam(params.all_touched, false)) {
        command << '--all-touched'
    }
    if (boolParam(params.no_predict, false)) {
        command << '--no-predict'
    }
    if (boolParam(params.fail_on_empty_labels, false)) {
        command << '--fail-on-empty-labels'
    }
    """
    export PYTHONPATH="${projectDir}/src:\${PYTHONPATH:-}"
    ${command.join(' ')} > prepare_tile_batch.json
    ${params.python} - <<'PY'
    import json
    from pathlib import Path

    payload = json.loads(Path("prepare_tile_batch.json").read_text(encoding="utf-8"))
    batch_manifest_path = Path(payload["batch_manifest_path"])
    Path("batch_manifest.json").write_text(batch_manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    Path("tile_manifest_paths.txt").write_text(
        "\\n".join(payload["tile_manifest_paths"]) + "\\n",
        encoding="utf-8",
    )
    PY
    """
}

process RUN_TILE_PLAN {
    label 'tile'
    tag "${tile_plan.simpleName}"

    input:
    path tile_plan

    output:
    path 'tile_result.json', emit: tile_results

    script:
    """
    export PYTHONPATH="${projectDir}/src:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"
    ${params.python} -m crome.cli run-tile-plan --tile-plan ${shellQuote(tile_plan.toString())} > tile_result.json
    """
}

process TRAIN_POOLED_MODEL {
    label 'pooled'
    tag "${runLabel}"

    input:
    path batch_manifest
    path tile_result_files

    output:
    path 'pooled_run.json', emit: pooled_summary

    script:
    def command = [
        "${params.python} -m crome.cli train-pooled-from-tile-results",
        "--batch-manifest ${shellQuote(batch_manifest.toString())}",
        "--test-size ${params.test_size}",
        "--random-state ${params.random_state}",
        "--n-estimators ${params.n_estimators}",
        "--n-jobs ${pooledNJobs}",
    ]
    if (params.pooled_output_dir) {
        command << "--output-dir ${shellQuote(params.pooled_output_dir)}"
    }
    if (params.max_train_rows != null) {
        command << "--max-train-rows ${params.max_train_rows}"
    }
    """
    export PYTHONPATH="${projectDir}/src:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"
    declare -a TILE_RESULT_ARGS=()
    while IFS= read -r result_path; do
      if [[ -n "${result_path}" ]]; then
        TILE_RESULT_ARGS+=( --tile-result "${result_path}" )
      fi
    done < <(cat ${tile_result_files.collect { shellQuote(it.toString()) }.join(' ')})
    ${command.join(' ')} "\${TILE_RESULT_ARGS[@]}" > pooled_run.json
    """
}

workflow {
    preparedBatch = PREPARE_TILE_BATCH()

    tilePlans = preparedBatch.out.tile_plan_paths
        .splitText()
        .filter { it?.trim() }
        .map { file(it.trim()) }

    tileRuns = RUN_TILE_PLAN(tilePlans)

    if (params.run_pooled_model) {
        TRAIN_POOLED_MODEL(preparedBatch.out.batch_manifest, tileRuns.out.tile_results.collect())
    }
}
