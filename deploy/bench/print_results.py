#!/usr/bin/env python3
"""Print a compact overview of cuvs-bench CSV results."""

import argparse
from pathlib import Path

import pandas as pd

_BUILD_COLUMNS = {"algo_name", "index_name", "time"}
_SEARCH_COLUMNS = {
    "algo_name",
    "index_name",
    "recall",
    "throughput",
    "latency",
}
_METADATA_COLUMNS = {
    "batch_size",
    "build time",
    "engine",
    "flush_time_seconds",
    "force_merge_time_seconds",
    "indexed_vectors",
    "ingest_threads",
    "ingest_time_seconds",
    "ingest_vectors_per_second",
    "num_batches",
    "refresh_time_seconds",
    "remote_build_wait_time_seconds",
    "space_type",
}

_BUILD_METADATA_COLUMNS = {
    "indexed_vectors": "vectors",
    "ingest_threads": "threads",
    "ingest_time_seconds": "ingest_s",
    "ingest_vectors_per_second": "vectors/s",
    "flush_time_seconds": "flush_s",
    "remote_build_wait_time_seconds": "remote_wait_s",
    "force_merge_time_seconds": "merge_s",
    "refresh_time_seconds": "refresh_s",
}


def _format_params(row, excluded: set[str]) -> str:
    values = []
    for name, value in row.items():
        if name in excluded or pd.isna(value):
            continue
        values.append(f"{name}={value}")
    return ", ".join(values) or "default"


def print_results(
    dataset_path: str,
    dataset: str,
    algorithm: str,
    groups: str,
    count: int,
    batch_size: int,
) -> None:
    result_dir = Path(dataset_path) / dataset / "result"
    group_names = [
        group.strip() for group in groups.split(",") if group.strip()
    ]
    build_metadata = []

    print("\nBuild results:")
    for group in group_names:
        build_file = result_dir / "build" / f"{algorithm},{group}.csv"
        if not build_file.exists():
            print(f"  [{group}] no build results")
            continue
        build = pd.read_csv(build_file)
        for _, row in build.iterrows():
            params = _format_params(
                row, _BUILD_COLUMNS | _METADATA_COLUMNS
            )
            print(
                f"  {row['algo_name']} index={row['index_name']} "
                f"time={float(row['time']):.2f}s params={params}"
            )

        metadata_columns = [
            column
            for column in _BUILD_METADATA_COLUMNS
            if column in build.columns
        ]
        if metadata_columns:
            metadata = build[["index_name", *metadata_columns]].copy()
            metadata.insert(0, "group", group)
            build_metadata.append(metadata)

    print("\nBuild metadata:")
    if not build_metadata:
        print("  no build metadata")
    else:
        metadata = pd.concat(build_metadata, ignore_index=True).rename(
            columns=_BUILD_METADATA_COLUMNS
        )
        formatters = {
            "vectors": lambda value: f"{int(value):,}",
            "threads": lambda value: f"{int(value)}",
            "vectors/s": lambda value: f"{float(value):,.1f}",
        }
        formatters.update(
            {
                column: lambda value: f"{float(value):.3f}"
                for column in metadata.columns
                if column.endswith("_s")
            }
        )
        print(metadata.to_string(index=False, formatters=formatters))

    print("\nSearch results:")
    for group in group_names:
        stem = f"{algorithm},{group},k{count},bs{batch_size},raw.csv"
        search_file = result_dir / "search" / stem
        if not search_file.exists():
            print(f"  [{group}] no search results")
            continue
        for _, row in pd.read_csv(search_file).iterrows():
            params = _format_params(
                row, _SEARCH_COLUMNS | _METADATA_COLUMNS
            )
            print(
                f"  {row['algo_name']} index={row['index_name']} "
                f"params={params} recall={float(row['recall']):.4f} "
                f"qps={float(row['throughput']):.1f} "
                f"latency={float(row['latency']) * 1000.0:.2f}ms"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    args = parser.parse_args()
    print_results(
        args.dataset_path,
        args.dataset,
        args.algorithm,
        args.groups,
        args.count,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
