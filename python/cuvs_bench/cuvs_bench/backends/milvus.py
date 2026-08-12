#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Milvus GPU benchmark backend."""

import os
import re
import time
from typing import Any

import numpy as np

from ..orchestrator.config_loaders import (
    BenchmarkConfig,
    ConfigLoader,
    DatasetConfig,
    IndexConfig,
)
from .base import BenchmarkBackend, BuildResult, SearchResult


class MilvusConfigLoader(ConfigLoader):
    """Load ``milvus_*`` algorithm configurations."""

    def __init__(self, config_path: str | os.PathLike | None = None):
        self.config_path = (
            os.fspath(config_path)
            if config_path
            else os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "../config"
            )
        )

    @property
    def backend_type(self) -> str:
        return "milvus"

    def _discover_algo_groups(
        self, dataset_conf, dataset, dataset_path, **kwargs
    ):
        files = [
            path
            for path in self.gather_algorithm_configs(
                self.config_path, kwargs.get("algorithm_configuration")
            )
            if os.path.basename(path).startswith("milvus_")
        ]
        allowed_algos = (
            {value.strip() for value in kwargs["algorithms"].split(",")}
            if kwargs.get("algorithms")
            else None
        )
        allowed_groups = (
            {value.strip() for value in kwargs["groups"].split(",")}
            if kwargs.get("groups")
            else None
        )
        result = []
        for path in files:
            config = self.load_yaml_file(path)
            name = config.get("name", "")
            if allowed_algos and name not in allowed_algos:
                continue
            for group, group_config in config.get("groups", {}).items():
                if not allowed_groups or group in allowed_groups:
                    result.append((name, group, group_config, {}))
        return result

    def _build_benchmark_configs(
        self,
        dataset_config: DatasetConfig,
        dataset_conf: dict,
        dataset: str,
        dataset_path: str,
        expanded_groups: list[tuple[str, str, dict, list, list, dict]],
        **kwargs,
    ) -> list[BenchmarkConfig]:
        connection = {
            key: kwargs[key]
            for key in ("host", "port", "uri", "token", "insert_batch_size")
            if key in kwargs
        }
        result = []
        for algo, group, _, build_combos, search_combos, _ in expanded_groups:
            for build_params in build_combos:
                prefix = algo if group == "base" else f"{algo}_{group}"
                label = ".".join(
                    [
                        prefix,
                        *(
                            f"{key}{value}"
                            for key, value in build_params.items()
                        ),
                    ]
                )
                collection = _safe_name(f"{dataset}_{label}")
                index = IndexConfig(
                    name=label,
                    algo=algo,
                    build_param=build_params,
                    search_params=search_combos,
                    file=collection,
                )
                result.append(
                    BenchmarkConfig(
                        indexes=[index],
                        backend_config={
                            "name": label,
                            "group": group,
                            "algo": algo,
                            "collection": collection,
                            "requires_network": True,
                            **connection,
                        },
                    )
                )
        return result


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", value)[:255]


_METRICS = {
    "euclidean": "L2",
    "l2": "L2",
    "inner_product": "IP",
    "innerproduct": "IP",
}


class MilvusBackend(BenchmarkBackend):
    """Benchmark Milvus GPU indexes through ``pymilvus.MilvusClient``."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.__client = None
        self.__data_type = None
        self._network_error = None

    @property
    def algo(self) -> str:
        return self.config.get("algo", "milvus_gpu_cagra")

    @property
    def _client(self):
        if self.__client is None:
            try:
                from pymilvus import DataType, MilvusClient
            except ImportError as error:
                raise ImportError(
                    "The Milvus backend requires pymilvus. Install cuvs-bench[milvus]."
                ) from error
            uri = self.config.get("uri") or (
                f"http://{self.config.get('host', 'localhost')}:"
                f"{self.config.get('port', 19530)}"
            )
            kwargs = {"uri": uri}
            if self.config.get("token"):
                kwargs["token"] = self.config["token"]
            self.__client = MilvusClient(**kwargs)
            self.__data_type = DataType
        return self.__client

    def initialize(self) -> None:
        # Keep the client lazy so CLI dry runs do not require a live service.
        pass

    def cleanup(self) -> None:
        if self.__client is not None:
            self.__client.close()
            self.__client = None

    def _check_network_available(self) -> bool:
        try:
            self._client.list_collections()
            self._network_error = None
            return True
        except ImportError:
            raise
        except Exception as error:  # noqa: BLE001 - errors vary by transport
            self._network_error = str(error)
            return False

    def _pre_flight_error(self, skip: str) -> str:
        message = f"pre-flight check failed: {skip}"
        if skip == "no_network" and self._network_error:
            message += f" ({self._network_error})"
        return message

    def _collection(self, index: IndexConfig) -> str:
        return self.config.get("collection", _safe_name(index.name))

    def build(
        self, dataset, indexes, force=False, dry_run=False
    ) -> BuildResult:
        if not indexes:
            return self._failed_build("No indexes provided")
        index = indexes[0]
        collection = self._collection(index)
        metric = _METRICS.get(dataset.distance_metric)
        if metric is None:
            return self._failed_build(
                f"Milvus GPU indexes do not support {dataset.distance_metric!r}",
                index.build_param,
            )
        if dry_run:
            print(
                f"[dry_run] Would build Milvus collection '{collection}' "
                f"with GPU_CAGRA and {index.build_param}"
            )
            return self._build_result(collection, index.build_param, 0.0, True)

        skip = self._pre_flight_check()
        if skip:
            return self._failed_build(self._pre_flight_error(skip))
        if self._client.has_collection(collection_name=collection):
            if not force:
                result = self._build_result(
                    collection, index.build_param, 0.0, True
                )
                result.metadata["skipped"] = True
                return result
            self._client.drop_collection(collection_name=collection)

        vectors = dataset.training_vectors
        if vectors.size == 0:
            return self._failed_build(
                "No training vectors available", index.build_param
            )

        start = time.perf_counter()
        schema = self._client.create_schema(
            auto_id=False, enable_dynamic_field=False
        )
        schema.add_field(
            field_name="id", datatype=self.__data_type.INT64, is_primary=True
        )
        schema.add_field(
            field_name="vector",
            datatype=self.__data_type.FLOAT_VECTOR,
            dim=vectors.shape[1],
        )
        self._client.create_collection(
            collection_name=collection,
            schema=schema,
            consistency_level="Strong",
        )
        batch_size = int(self.config.get("insert_batch_size", 1000))
        for offset in range(0, len(vectors), batch_size):
            self._client.insert(
                collection_name=collection,
                data=[
                    {"id": row, "vector": vector.tolist()}
                    for row, vector in enumerate(
                        vectors[offset : offset + batch_size], start=offset
                    )
                ],
            )
        self._client.flush(collection_name=collection)
        params = self._client.prepare_index_params()
        params.add_index(
            field_name="vector",
            index_type="GPU_CAGRA",
            metric_type=metric,
            params=index.build_param,
        )
        self._client.create_index(
            collection_name=collection, index_params=params, sync=True
        )
        self._client.load_collection(collection_name=collection)
        return self._build_result(
            collection, index.build_param, time.perf_counter() - start, True
        )

    def search(
        self,
        dataset,
        indexes,
        k,
        batch_size=10000,
        mode="latency",
        force=False,
        search_threads=None,
        dry_run=False,
    ) -> list[SearchResult]:
        if not indexes:
            return [self._failed_search(k, "No indexes provided")]
        index = indexes[0]
        collection = self._collection(index)
        combinations = index.search_params or [{}]
        if dry_run:
            print(
                f"[dry_run] Would search Milvus collection '{collection}' "
                f"with {len(combinations)} parameter set(s)"
            )
            return [
                self._search_result(k, collection, params)
                for params in combinations
            ]
        skip = self._pre_flight_check()
        if skip:
            return [self._failed_search(k, self._pre_flight_error(skip))]

        queries = dataset.query_vectors
        if queries.size == 0:
            return [self._failed_search(k, "No query vectors available")]
        metric = _METRICS.get(dataset.distance_metric)
        if metric is None:
            return [
                self._failed_search(
                    k,
                    f"Milvus GPU indexes do not support {dataset.distance_metric!r}",
                )
            ]
        results = []
        for params in combinations:
            neighbors = np.full((len(queries), k), -1, dtype=np.int64)
            distances = np.full((len(queries), k), np.nan, dtype=np.float32)
            start = time.perf_counter()
            batches = 0
            for offset in range(0, len(queries), batch_size):
                response = self._client.search(
                    collection_name=collection,
                    data=queries[offset : offset + batch_size].tolist(),
                    anns_field="vector",
                    search_params={"metric_type": metric, "params": params},
                    limit=k,
                )
                for row, hits in enumerate(response, start=offset):
                    for column, hit in enumerate(hits[:k]):
                        neighbors[row, column] = int(hit["id"])
                        distances[row, column] = float(hit["distance"])
                batches += 1
            elapsed = time.perf_counter() - start
            results.append(
                SearchResult(
                    neighbors=neighbors,
                    distances=distances,
                    search_time_ms=elapsed * 1000,
                    queries_per_second=len(queries) / elapsed
                    if elapsed
                    else 0.0,
                    recall=0.0,
                    algorithm=self.algo,
                    search_params=[params],
                    metadata={
                        "group": self.config["group"],
                        "index_name": collection,
                        "latency_seconds": elapsed / batches,
                    },
                )
            )
        return results

    def _build_result(self, collection, params, elapsed, success):
        return BuildResult(
            index_path=collection,
            build_time_seconds=elapsed,
            index_size_bytes=0,
            algorithm=self.algo,
            build_params=params,
            metadata={
                "group": self.config["group"],
                "index_type": "GPU_CAGRA",
            },
            success=success,
        )

    def _failed_build(self, message, params=None):
        result = self._build_result(
            self.config.get("collection", ""), params or {}, 0.0, False
        )
        result.error_message = message
        return result

    def _search_result(self, k, collection, params):
        return SearchResult(
            neighbors=np.empty((0, k), dtype=np.int64),
            distances=np.empty((0, k), dtype=np.float32),
            search_time_ms=0.0,
            queries_per_second=0.0,
            recall=0.0,
            algorithm=self.algo,
            search_params=[params],
            metadata={"group": self.config["group"], "index_name": collection},
        )

    def _failed_search(self, k, message):
        result = self._search_result(k, self.config.get("collection", ""), {})
        result.success = False
        result.error_message = message
        return result
