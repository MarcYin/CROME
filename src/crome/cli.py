"""Top-level CLI for package-safe migration commands."""

from __future__ import annotations

import argparse
import sys

from crome.acquisition import alphaearth, crome
from crome import discovery, labeling, orchestration, pipeline, predict, training, workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crome", description="CROME migration utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    alphaearth_parser = alphaearth.build_parser()
    alphaearth_help = alphaearth_parser.description or "Download AlphaEarth annual embeddings."
    subparsers.add_parser(
        "download-alphaearth",
        help=alphaearth_help,
        parents=[alphaearth_parser],
        add_help=False,
    )
    crome_parser = crome.build_parser()
    crome_help = crome_parser.description or "Download a CROME GeoPackage reference."
    subparsers.add_parser(
        "download-crome",
        help=crome_help,
        parents=[crome_parser],
        add_help=False,
    )
    crome_subset_parser = crome.build_subset_parser()
    subparsers.add_parser(
        "prepare-crome-subset",
        help=(
            crome_subset_parser.description
            or "Materialize or reuse one AOI-specific CROME subset for discovered AlphaEarth tiles."
        ),
        parents=[crome_subset_parser],
        add_help=False,
    )
    discovery_parser = discovery.build_parser()
    subparsers.add_parser(
        "list-feature-rasters",
        help=discovery_parser.description or "List discovered AlphaEarth feature rasters.",
        parents=[discovery_parser],
        add_help=False,
    )
    subparsers.add_parser(
        "discover-feature-rasters",
        help=discovery_parser.description or "List discovered AlphaEarth feature rasters.",
        parents=[discovery_parser],
        add_help=False,
    )
    rasterize_parser = labeling.build_parser()
    subparsers.add_parser(
        "rasterize-reference",
        help=rasterize_parser.description or "Rasterize CROME references onto the AlphaEarth grid.",
        parents=[rasterize_parser],
        add_help=False,
    )
    training_table_parser = training.build_training_table_parser()
    subparsers.add_parser(
        "build-training-table",
        help=training_table_parser.description or "Build a training table from aligned rasters.",
        parents=[training_table_parser],
        add_help=False,
    )
    training_table_from_cache_parser = training.build_training_table_from_cache_parser()
    subparsers.add_parser(
        "build-training-table-from-cache",
        help=(
            training_table_from_cache_parser.description
            or "Build a training table from one or more cached sample manifests."
        ),
        parents=[training_table_from_cache_parser],
        add_help=False,
    )
    pooled_model_parser = training.build_train_pooled_model_parser()
    subparsers.add_parser(
        "train-pooled-model",
        help=pooled_model_parser.description or "Build and train one pooled model from prior pipeline manifests.",
        parents=[pooled_model_parser],
        add_help=False,
    )
    train_model_parser = training.build_train_model_parser()
    subparsers.add_parser(
        "train-model",
        help=train_model_parser.description or "Train a crop classifier from a training table.",
        parents=[train_model_parser],
        add_help=False,
    )
    predict_parser = predict.build_parser()
    subparsers.add_parser(
        "predict-map",
        help=predict_parser.description or "Predict a crop map from an AlphaEarth feature raster.",
        parents=[predict_parser],
        add_help=False,
    )
    pipeline_parser = pipeline.build_parser()
    subparsers.add_parser(
        "run-baseline-pipeline",
        help=pipeline_parser.description or "Run the batch AlphaEarth-to-CROME baseline pipeline.",
        parents=[pipeline_parser],
        add_help=False,
    )
    workflow_parser = workflow.build_parser()
    subparsers.add_parser(
        "download-run-baseline",
        help=workflow_parser.description or "Download AlphaEarth annual embeddings and run the baseline pipeline.",
        parents=[workflow_parser],
        add_help=False,
    )
    prepare_tile_batch_parser = orchestration.build_prepare_tile_batch_parser()
    subparsers.add_parser(
        "prepare-tile-batch",
        help=prepare_tile_batch_parser.description or "Prepare one cluster-parallel tile batch.",
        parents=[prepare_tile_batch_parser],
        add_help=False,
    )
    run_tile_plan_parser = orchestration.build_run_tile_plan_parser()
    subparsers.add_parser(
        "run-tile-plan",
        help=run_tile_plan_parser.description or "Run one prepared per-tile plan.",
        parents=[run_tile_plan_parser],
        add_help=False,
    )
    pooled_from_tile_results_parser = orchestration.build_train_pooled_from_tile_results_parser()
    subparsers.add_parser(
        "train-pooled-from-tile-results",
        help=(
            pooled_from_tile_results_parser.description
            or "Train one pooled model from prepared tile result JSON payloads."
        ),
        parents=[pooled_from_tile_results_parser],
        add_help=False,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    forwarded = argv[1:] if argv is not None else sys.argv[2:]
    if args.command == "download-alphaearth":
        return alphaearth.main(forwarded)
    if args.command == "download-crome":
        return crome.main(forwarded)
    if args.command == "prepare-crome-subset":
        return crome.main_prepare_subset(forwarded)
    if args.command == "list-feature-rasters":
        return discovery.main(forwarded)
    if args.command == "discover-feature-rasters":
        return discovery.main(forwarded)
    if args.command == "rasterize-reference":
        return labeling.main(forwarded)
    if args.command == "build-training-table":
        return training.main_build_training_table(forwarded)
    if args.command == "build-training-table-from-cache":
        return training.main_build_training_table_from_cache(forwarded)
    if args.command == "train-pooled-model":
        return training.main_train_pooled_model(forwarded)
    if args.command == "train-model":
        return training.main_train_model(forwarded)
    if args.command == "predict-map":
        return predict.main(forwarded)
    if args.command == "run-baseline-pipeline":
        return pipeline.main(forwarded)
    if args.command == "download-run-baseline":
        return workflow.main(forwarded)
    if args.command == "prepare-tile-batch":
        return orchestration.main_prepare_tile_batch(forwarded)
    if args.command == "run-tile-plan":
        return orchestration.main_run_tile_plan(forwarded)
    if args.command == "train-pooled-from-tile-results":
        return orchestration.main_train_pooled_from_tile_results(forwarded)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
