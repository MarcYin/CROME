nextflow.enable.dsl = 2

import groovy.json.JsonSlurper

params.python = params.python ?: 'python'
params.project_src = params.project_src ?: "${projectDir}/src"
params.batch_manifest = null
params.run_pooled_model = true
params.pooled_output_dir = null
params.max_train_rows = null
params.n_estimators = null
params.random_state = null
params.test_size = null

def loadBatchManifest(manifestPath) {
    def path = file(manifestPath)
    if (!path.exists()) {
        error "Batch manifest does not exist: ${manifestPath}"
    }
    def payload = new JsonSlurper().parseText(path.text)
    if (!(payload instanceof Map)) {
        error "Batch manifest is not a JSON object: ${manifestPath}"
    }
    return payload
}

process RUN_TILE_PLAN {
    label 'tile_baseline'
    tag { tilePlan.baseName }

    input:
    path tilePlan

    output:
    path "${tilePlan.baseName}.tile-result.json"

    script:
    """
    export PYTHONPATH="${params.project_src}:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"
    ${params.python} -m crome.cli run-tile-plan --tile-plan "${tilePlan}" > "${tilePlan.baseName}.tile-result.json"
    """
}

process TRAIN_POOLED_MODEL {
    label 'pooled_model'
    tag { batchManifest.baseName }

    input:
    path batchManifest
    path tileResults

    output:
    path "pooled-model.result.json"

    script:
    def tileArgs = tileResults.collect { "--tile-result ${it}" }.join(' \\\n      ')
    def outputArg = params.pooled_output_dir ? "--output-dir ${params.pooled_output_dir}" : ""
    def maxTrainRowsArg = params.max_train_rows ? "--max-train-rows ${params.max_train_rows}" : ""
    def estimatorArg = params.n_estimators ? "--n-estimators ${params.n_estimators}" : ""
    def randomStateArg = params.random_state ? "--random-state ${params.random_state}" : ""
    def testSizeArg = params.test_size ? "--test-size ${params.test_size}" : ""
    """
    export PYTHONPATH="${params.project_src}:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"
    ${params.python} -m crome.cli train-pooled-from-tile-results \\
      --batch-manifest "${batchManifest}" \\
      ${tileArgs} \\
      ${outputArg} \\
      ${maxTrainRowsArg} \\
      ${estimatorArg} \\
      ${randomStateArg} \\
      ${testSizeArg} \\
      > pooled-model.result.json
    """
}

workflow {
    if (!params.batch_manifest) {
        error "Provide --batch_manifest /path/to/batch_manifest.json"
    }

    def batchManifestFile = file(params.batch_manifest)
    def batchPayload = loadBatchManifest(batchManifestFile)
    def tileManifestPaths = batchPayload.tile_manifest_paths ?: []
    if (!tileManifestPaths) {
        error "Batch manifest does not contain tile_manifest_paths: ${batchManifestFile}"
    }

    tileManifestChannel = Channel
        .fromList(tileManifestPaths.collect { file(it.toString()) })
        .ifEmpty { error "No tile manifests were resolved from ${batchManifestFile}" }

    tileResults = RUN_TILE_PLAN(tileManifestChannel)

    if (params.run_pooled_model) {
        TRAIN_POOLED_MODEL(Channel.value(batchManifestFile), tileResults.collect())
    }
}
