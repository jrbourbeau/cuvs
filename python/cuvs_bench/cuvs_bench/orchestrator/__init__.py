#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from ..backends.milvus import MilvusConfigLoader
from ..backends.opensearch import OpenSearchConfigLoader
from ..backends.registry import (
    get_backend_class,
    get_config_loader,
    list_backends,
    register_config_loader,
)
from .config_loaders import (
    BenchmarkConfig,
    ConfigLoader,
    CppGBenchConfigLoader,
    DatasetConfig,
)
from .orchestrator import BenchmarkOrchestrator

__all__ = [
    "BenchmarkConfig",
    "BenchmarkOrchestrator",
    "ConfigLoader",
    "CppGBenchConfigLoader",
    "DatasetConfig",
    "MilvusConfigLoader",
    "OpenSearchConfigLoader",
    "get_backend_class",
    "get_config_loader",
    "list_backends",
    "register_config_loader",
]


# ============================================================================
# Register built-in config loaders
# ============================================================================


def _register_builtin_loaders():
    """Register built-in config loaders."""
    register_config_loader("cpp_gbench", CppGBenchConfigLoader)
    register_config_loader("milvus", MilvusConfigLoader)
    register_config_loader("opensearch", OpenSearchConfigLoader)


# Auto-register when module is imported
_register_builtin_loaders()
