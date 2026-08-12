#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
cuvs-bench backends package.

This package provides the plugin architecture for benchmarking various
vector database backends, including C++ executables, Python libraries,
and network-based VDB services.
"""

from .base import (
    BenchmarkBackend,
    BuildResult,
    Dataset,
    SearchResult,
)
from .cpp_gbench import CppGoogleBenchmarkBackend
from .milvus import MilvusBackend
from .opensearch import OpenSearchBackend
from .registry import (
    BackendRegistry,
    get_backend,
    get_registry,
    register_backend,
)

# Auto-register built-in backends
_registry = get_registry()
_registry.register("cpp_gbench", CppGoogleBenchmarkBackend)
_registry.register("milvus", MilvusBackend)
_registry.register("opensearch", OpenSearchBackend)

__all__ = [
    "BackendRegistry",
    "BenchmarkBackend",
    "BuildResult",
    "CppGoogleBenchmarkBackend",
    "Dataset",
    "MilvusBackend",
    "OpenSearchBackend",
    "SearchResult",
    "get_backend",
    "get_registry",
    "register_backend",
]
