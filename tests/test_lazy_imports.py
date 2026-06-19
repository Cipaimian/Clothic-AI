"""The heavy perception modules must IMPORT without torch/ultralytics.

They load their dependencies lazily (inside __init__/methods), so the base
package stays light and CI runs without GPUs. Instantiating them would require
the optional deps; importing them must not.
"""

from __future__ import annotations

import importlib

import pytest

HEAVY_MODULES = [
    "clothic.perception.pose",
    "clothic.perception.parsing",
    "clothic.perception.attributes",
    "clothic.perception.full_backend",
    "clothic.perception.ultralytics_backend",
]


@pytest.mark.parametrize("mod", HEAVY_MODULES)
def test_heavy_module_imports_without_torch(mod):
    m = importlib.import_module(mod)
    assert m is not None


def test_factory_falls_back_to_mock_without_deps():
    import importlib.util

    if importlib.util.find_spec("ultralytics") is not None:
        pytest.skip("real deps installed; factory builds the real backend, no fallback")

    from clothic.perception import get_backend
    from clothic.perception.mock_backend import MockBackend

    # 'full' needs torch; absent it, the factory must degrade to the mock backend.
    backend = get_backend("full")
    assert isinstance(backend, MockBackend)
