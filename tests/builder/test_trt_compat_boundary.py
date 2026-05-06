from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

from tensorrt_model_connect import trt_compat


REPO_ROOT = Path(__file__).resolve().parents[2]
TRTMC_BUILD_ROOT = REPO_ROOT / "tensorrt_model_connect" / "tensorrt_model_connect"
ALLOWED_TRT_BOUNDARY_FILES = {
    TRTMC_BUILD_ROOT / "trt_compat.py",
}


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_importlib_import_module(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "import_module"
        and isinstance(node.value, ast.Name)
        and node.value.id == "importlib"
    )


def _is_sys_modules_subscript(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "modules"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "sys"
        and _constant_string(node.slice) in {"tensorrt", "tensorrt_rtx"}
    )


def test_tensor_rt_python_api_is_imported_only_through_compat_layer():
    """Builder code must route TensorRT Python API access through trt_compat."""
    violations: list[str] = []
    for path in sorted(TRTMC_BUILD_ROOT.rglob("*.py")):
        if path in ALLOWED_TRT_BOUNDARY_FILES:
            continue
        rel = path.relative_to(REPO_ROOT)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {"tensorrt", "tensorrt_rtx"}:
                        violations.append(f"{rel}:{node.lineno} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module in {"tensorrt", "tensorrt_rtx"}:
                    violations.append(f"{rel}:{node.lineno} imports from {node.module}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                    if node.args and _constant_string(node.args[0]) in {"tensorrt", "tensorrt_rtx"}:
                        violations.append(f"{rel}:{node.lineno} dynamically imports TensorRT")
                elif _is_importlib_import_module(node.func):
                    if node.args and _constant_string(node.args[0]) in {"tensorrt", "tensorrt_rtx"}:
                        violations.append(f"{rel}:{node.lineno} dynamically imports TensorRT")
            elif _is_sys_modules_subscript(node) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                violations.append(f"{rel}:{node.lineno} mutates sys.modules TensorRT alias")
            elif isinstance(node, ast.Attribute) and node.attr == "EXPLICIT_BATCH":
                violations.append(f"{rel}:{node.lineno} uses EXPLICIT_BATCH directly")

    assert violations == []


def test_trt_compat_proxy_wraps_version_sensitive_builder_calls(monkeypatch):
    """The lazy trt proxy intercepts builder/network/config calls with a fake TRT module."""
    calls: list[tuple] = []

    class FakeLogger:
        WARNING = 1
        VERBOSE = 2

        def __init__(self, level):
            self.level = level

    class FakeNetwork:
        def __init__(self, flags):
            self.flags = flags

        def add_matrix_multiply(self, lhs, lhs_op, rhs, rhs_op):
            calls.append(("add_matrix_multiply", lhs, lhs_op, rhs, rhs_op))
            return "layer"

    class FakeConfig:
        def set_memory_pool_limit(self, pool, size):
            calls.append(("set_memory_pool_limit", pool, size))

    class FakeBuilder:
        def __init__(self, logger):
            self.logger = logger

        def create_network(self, flags=0):
            calls.append(("create_network", flags))
            return FakeNetwork(flags)

        def create_builder_config(self):
            return FakeConfig()

        def build_serialized_network(self, network, config):
            assert isinstance(network, FakeNetwork)
            assert isinstance(config, FakeConfig)
            return b"plan"

    class FakeRuntime:
        pass

    fake_trt = types.ModuleType("tensorrt")
    fake_trt.__version__ = "10.16.1.11"
    fake_trt.Logger = FakeLogger
    fake_trt.Builder = FakeBuilder
    fake_trt.Runtime = FakeRuntime
    fake_trt.IBuilder = FakeBuilder
    fake_trt.INetworkDefinition = FakeNetwork
    fake_trt.IBuilderConfig = FakeConfig
    fake_trt.IRuntime = FakeRuntime
    fake_trt.MemoryPoolType = types.SimpleNamespace(WORKSPACE="workspace")
    fake_trt.NetworkDefinitionCreationFlag = types.SimpleNamespace(STRONGLY_TYPED=1)

    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.setattr(trt_compat, "_module", None)
    monkeypatch.setattr(trt_compat, "_backend_module_name", "tensorrt")
    monkeypatch.setattr(trt_compat, "_backend_label", "TensorRT")

    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = trt_compat.network_creation_flags(strongly_typed=True, explicit_batch=True)
    network = builder.create_network(flags)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024)
    layer = network.add_matrix_multiply("lhs", "none", "rhs", "transpose")
    plan = builder.build_serialized_network(network, config)

    assert flags == 2
    assert layer == "layer"
    assert plan == b"plan"
    assert calls == [
        ("create_network", 2),
        ("set_memory_pool_limit", "workspace", 1024),
        ("add_matrix_multiply", "lhs", "none", "rhs", "transpose"),
    ]
