#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
from cuvs_bench.backends.base import Dataset
from cuvs_bench.backends.milvus import MilvusBackend, MilvusConfigLoader
from cuvs_bench.backends.registry import get_registry
from cuvs_bench.orchestrator.config_loaders import IndexConfig


class _Builder:
    def __init__(self):
        self.fields = []
        self.indexes = []

    def add_field(self, **kwargs):
        self.fields.append(kwargs)

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class _Client:
    def __init__(self):
        self.calls = []

    def list_collections(self):
        return []

    def has_collection(self, **kwargs):
        return False

    def create_schema(self, **kwargs):
        self.calls.append(("create_schema", kwargs))
        return _Builder()

    def create_collection(self, **kwargs):
        self.calls.append(("create_collection", kwargs))

    def insert(self, **kwargs):
        self.calls.append(("insert", kwargs))

    def flush(self, **kwargs):
        self.calls.append(("flush", kwargs))

    def compact(self, **kwargs):
        self.calls.append(("compact", kwargs))
        return 42

    def get_compaction_state(self, **kwargs):
        self.calls.append(("get_compaction_state", kwargs))
        return "Completed"

    def prepare_index_params(self):
        self.index_params = _Builder()
        return self.index_params

    def create_index(self, **kwargs):
        self.calls.append(("create_index", kwargs))

    def load_collection(self, **kwargs):
        self.calls.append(("load_collection", kwargs))

    def release_collection(self, **kwargs):
        self.calls.append(("release_collection", kwargs))

    def close(self):
        self.calls.append(("close", {}))

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        return [
            [{"id": row, "distance": float(row)}]
            for row in range(len(kwargs["data"]))
        ]


class _UnavailableClient:
    def list_collections(self):
        raise RuntimeError("connection refused")


def _backend():
    backend = MilvusBackend(
        {
            "name": "test",
            "group": "test",
            "algo": "milvus_gpu_cagra",
            "collection": "test_collection",
            "requires_network": True,
            "insert_batch_size": 2,
        }
    )
    client = _Client()
    backend._MilvusBackend__client = client
    backend._MilvusBackend__data_type = SimpleNamespace(
        INT64="INT64", FLOAT_VECTOR="FLOAT_VECTOR"
    )
    return backend, client


def _dataset():
    return Dataset(
        name="test",
        training_vectors=np.arange(20, dtype=np.float32).reshape(5, 4),
        query_vectors=np.arange(12, dtype=np.float32).reshape(3, 4),
        groundtruth_neighbors=np.zeros((3, 1), dtype=np.int64),
        distance_metric="euclidean",
    )


def _index():
    return IndexConfig(
        name="test",
        algo="milvus_gpu_cagra",
        build_param={"intermediate_graph_degree": 64, "graph_degree": 32},
        search_params=[{"itopk_size": 64, "search_width": 1}],
        file="test_collection",
    )


def test_milvus_is_registered():
    assert get_registry().is_registered("milvus")


def test_config_loader(tmp_path):
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "datasets.yaml").write_text(
        "- name: test-ds\n  distance: euclidean\n  dims: 4\n"
    )
    (tmp_path / "algos").mkdir()
    (tmp_path / "algos" / "milvus_gpu_cagra.yaml").write_text(
        """name: milvus_gpu_cagra
groups:
  test:
    build: {graph_degree: [32]}
    search: {search_width: [1, 2]}
"""
    )
    _, configs = MilvusConfigLoader(tmp_path).load(
        dataset="test-ds", dataset_path="/data", groups="test", port=19531
    )
    assert len(configs) == 1
    assert configs[0].backend_config["port"] == 19531
    assert len(configs[0].indexes[0].search_params) == 2


def test_build_creates_gpu_cagra_index():
    backend, client = _backend()
    result = backend.build(_dataset(), [_index()])
    assert result.success
    assert result.index_path == "test_collection"
    assert client.index_params.indexes[0]["index_type"] == "GPU_CAGRA"
    assert len([call for call in client.calls if call[0] == "insert"]) == 3
    calls = [call[0] for call in client.calls]
    assert calls.index("flush") < calls.index("compact")
    assert calls.index("compact") < calls.index("get_compaction_state")
    assert calls.index("get_compaction_state") < calls.index("create_index")
    assert client.calls[calls.index("compact")][1]["target_size"] == (
        1 << 63
    ) - 1


def test_search_returns_neighbor_arrays():
    backend, client = _backend()
    result = backend.search(_dataset(), [_index()], k=1, batch_size=2)[0]
    assert result.success
    assert result.neighbors.shape == (3, 1)
    search_calls = [call for call in client.calls if call[0] == "search"]
    assert len(search_calls) == 2
    assert client.calls.index(search_calls[0]) > [
        call[0] for call in client.calls
    ].index("load_collection")
    assert search_calls[0][1]["search_params"] == {
        "metric_type": "L2",
        "params": {"itopk_size": 64, "search_width": 1},
    }


def test_cleanup_releases_loaded_collection():
    backend, client = _backend()
    backend.build(_dataset(), [_index()])
    backend.cleanup()
    assert client.calls[-2:] == [
        ("release_collection", {"collection_name": "test_collection"}),
        ("close", {}),
    ]


def test_dry_run_does_not_connect():
    backend = MilvusBackend(
        {"name": "test", "group": "test", "collection": "test_collection"}
    )
    assert backend.build(_dataset(), [_index()], dry_run=True).success
    assert backend.search(_dataset(), [_index()], k=1, dry_run=True)[0].success


def test_network_error_is_reported():
    backend, _ = _backend()
    backend._MilvusBackend__client = _UnavailableClient()
    result = backend.build(_dataset(), [_index()])
    assert not result.success
    assert result.error_message == (
        "pre-flight check failed: no_network (connection refused)"
    )
