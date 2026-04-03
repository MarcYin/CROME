nextflow.enable.dsl = 2

/*
 * Full England CROME pipeline: download, prepare, train, predict for 2017-2024.
 *
 * Usage:
 *   # Dry-run locally
 *   nextflow run nextflow/england.nf -profile local --years '2024' --dry_run true
 *
 *   # Full run on JASMIN
 *   nextflow run nextflow/england.nf -c nextflow/nextflow.config -profile jasmin \
 *     --output_root /gws/ssde/j25a/nceo_isp/public/CROME \
 *     --slurm_account nceo_isp
 *
 *   # Resume after failure
 *   nextflow run nextflow/england.nf -c nextflow/nextflow.config -profile jasmin \
 *     --output_root /gws/ssde/j25a/nceo_isp/public/CROME \
 *     --slurm_account nceo_isp -resume
 */

import groovy.json.JsonSlurper
import groovy.json.JsonOutput

params.python = params.python ?: 'python'
def projectSrc = params.project_src ?: "${projectDir}/src"

// England CROME footprint bbox (EPSG:4326)
params.bbox = params.bbox ?: '-7.06,49.86,2.08,55.82'
params.years = params.years ?: '2017,2018,2019,2020,2021,2022,2023,2024'
params.output_root = params.output_root ?: './outputs'
params.download_workers = params.download_workers ?: 4
params.n_estimators = params.n_estimators ?: 400
params.run_pooled_model = params.run_pooled_model != null ? params.run_pooled_model : true
params.dry_run = params.dry_run ?: false

def tileNJobs = params.tile_n_jobs != null ? params.tile_n_jobs : params.tile_cpus
def pooledNJobs = params.pooled_n_jobs != null ? params.pooled_n_jobs : params.pooled_cpus

def pythonEnv = """
export PYTHONPATH="${projectSrc}:\${PYTHONPATH:-}"
export CROME_DATA_ROOT="${params.output_root}"
export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-\${task.cpus}}"
export OPENBLAS_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-\${task.cpus}}"
export MKL_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-\${task.cpus}}"
export NUMEXPR_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-\${task.cpus}}"
"""


/*
 * Step 1: Download CROME reference for one year.
 * Lightweight — runs on login node or small allocation.
 */
process DOWNLOAD_CROME {
    tag { "crome-${year}" }
    label 'download'

    input:
    val year

    output:
    tuple val(year), env(REF_PATH), emit: crome_refs

    script:
    """
    ${pythonEnv}
    REF_PATH=\$(${params.python} -c "
import json
from crome.acquisition.crome import download_crome_reference
from crome.config import CromeDownloadRequest
req = CromeDownloadRequest(year=${year}, output_root='${params.output_root}', prefer_complete=True, extract=True, force=False)
result = download_crome_reference(req)
print(result.reference_path)
")
    """
}


/*
 * Step 2: Download AlphaEarth tiles for one year's bbox.
 * I/O-bound — benefits from parallel workers but no heavy CPU.
 */
process DOWNLOAD_ALPHAEARTH {
    tag { "ae-${year}" }
    label 'download'

    input:
    tuple val(year), val(ref_path)

    output:
    tuple val(year), val(ref_path), path("download_result.json"), emit: ae_downloads

    script:
    def bboxParts = params.bbox.split(',')
    """
    ${pythonEnv}
    ${params.python} -c "
import json
from crome.config import AlphaEarthDownloadRequest
from crome.acquisition.alphaearth import download_alphaearth_images
req = AlphaEarthDownloadRequest(
    year=${year},
    output_root='${params.output_root}',
    aoi_label='england-crome-${year}',
    bbox=(${bboxParts[0]}, ${bboxParts[1]}, ${bboxParts[2]}, ${bboxParts[3]}),
)
result = download_alphaearth_images(req, download_workers=${params.download_workers}, prepare_workers=${params.download_workers})
json.dump({
    'manifest_path': str(result.manifest_path),
    'output_root': str(result.output_root),
    'image_count': len(result.source_image_ids),
    'year': ${year},
}, open('download_result.json', 'w'), indent=2)
print(f'Downloaded {len(result.source_image_ids)} images for year ${year}')
"
    """
}


/*
 * Step 3: Prepare tile batch from downloaded tiles.
 * Fast — reads manifests and writes JSON plans.
 */
process PREPARE_TILE_BATCH {
    tag { "batch-${year}" }
    label 'download'

    input:
    tuple val(year), val(ref_path), path(download_result)

    output:
    tuple val(year), path("batch_result.json"), emit: batches

    script:
    """
    ${pythonEnv}
    DOWNLOAD_INFO=\$(cat ${download_result})
    MANIFEST_PATH=\$(echo \$DOWNLOAD_INFO | ${params.python} -c "import json,sys; print(json.load(sys.stdin)['manifest_path'])")

    ${params.python} -c "
import json
from crome.orchestration import prepare_tile_batch
result = prepare_tile_batch(
    feature_input=None,
    manifest_path='\$MANIFEST_PATH',
    reference_path='${ref_path}',
    year=${year},
    output_root='${params.output_root}',
    aoi_label='england-crome-${year}',
    n_estimators=${params.n_estimators},
)
json.dump({
    'batch_manifest_path': str(result.batch_manifest_path),
    'tile_count': len(result.tile_manifest_paths),
    'tile_manifest_paths': [str(p) for p in result.tile_manifest_paths],
    'workflow_output_root': str(result.workflow_output_root),
    'year': ${year},
}, open('batch_result.json', 'w'), indent=2)
print(f'Prepared {len(result.tile_manifest_paths)} tile plans for year ${year}')
"
    """
}


/*
 * Step 4: Run one tile plan (label, train, predict).
 * CPU-heavy — gets a proper Slurm allocation.
 */
process RUN_TILE_PLAN {
    tag { tilePlan.baseName }
    label 'tile'

    input:
    tuple val(year), path(tilePlan)

    output:
    tuple val(year), path("${tilePlan.baseName}.tile-result.json"), emit: tile_results

    script:
    def nJobsArg = tileNJobs != null ? "--n-jobs ${tileNJobs}" : ""
    """
    ${pythonEnv}
    ${params.python} -m crome.cli run-tile-plan --tile-plan "${tilePlan}" ${nJobsArg} > "${tilePlan.baseName}.tile-result.json"
    """
}


/*
 * Step 5: Train pooled model from all tile results for one year.
 * CPU-heavy — gets a large Slurm allocation.
 */
process TRAIN_POOLED_MODEL {
    tag { "pooled-${year}" }
    label 'pooled'

    input:
    tuple val(year), path(batchManifest), val(tileResultPaths)

    output:
    tuple val(year), path("pooled-${year}.result.json"), emit: pooled_results

    script:
    def tileArgs = tileResultPaths.collect { "--tile-result \"${it}\"" }.join(' \\\n      ')
    def nJobsArg = pooledNJobs != null ? "--n-jobs ${pooledNJobs}" : ""
    def maxTrainRowsArg = params.max_train_rows ? "--max-train-rows ${params.max_train_rows}" : ""
    """
    ${pythonEnv}
    ${params.python} -m crome.cli train-pooled-from-tile-results \\
      --batch-manifest "${batchManifest}" \\
      ${tileArgs} \\
      ${nJobsArg} \\
      ${maxTrainRowsArg} \\
      > "pooled-${year}.result.json"
    """
}


/*
 * Main workflow: all years run in parallel end-to-end.
 *
 * DOWNLOAD_CROME -> DOWNLOAD_ALPHAEARTH -> PREPARE_TILE_BATCH
 *   -> fan-out RUN_TILE_PLAN per tile -> TRAIN_POOLED_MODEL per year
 *
 * -resume picks up at exactly the step that failed.
 */
workflow {
    yearsList = params.years.toString().split(',').collect { it.trim() as int }

    // Step 1: Download CROME references (all years in parallel)
    yearChannel = Channel.fromList(yearsList)
    cromeRefs = DOWNLOAD_CROME(yearChannel)

    // Step 2: Download AlphaEarth tiles (all years in parallel)
    aeDownloads = DOWNLOAD_ALPHAEARTH(cromeRefs)

    // Step 3: Prepare tile batches
    batches = PREPARE_TILE_BATCH(aeDownloads)

    // Step 4: Fan out tiles from each batch
    tilePlans = batches.flatMap { year, batchResult ->
        def payload = new JsonSlurper().parseText(batchResult.text)
        def tilePaths = payload.tile_manifest_paths ?: []
        tilePaths.collect { tilePath -> tuple(year, file(tilePath)) }
    }

    tileResults = RUN_TILE_PLAN(tilePlans)

    // Step 5: Pooled training per year
    if (params.run_pooled_model) {
        // Group tile results by year, pair with batch manifest
        groupedResults = tileResults
            .map { year, resultFile -> tuple(year, resultFile.toString()) }
            .groupTuple()

        batchManifests = batches.map { year, batchResult ->
            def payload = new JsonSlurper().parseText(batchResult.text)
            tuple(year, file(payload.batch_manifest_path))
        }

        pooledInputs = batchManifests
            .join(groupedResults)
            .map { year, batchManifest, tileResultPaths ->
                tuple(year, batchManifest, tileResultPaths)
            }

        TRAIN_POOLED_MODEL(pooledInputs)
    }
}
