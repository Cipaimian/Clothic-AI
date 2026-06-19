"""Perception core: turns frames into ``PersonObservation`` objects.

Backends are interchangeable. ``MockBackend`` runs anywhere with zero heavy
deps (great for developing the reasoning core, tests, and the API). The
``UltralyticsBackend`` activates real detection when ``ultralytics``/``torch``
are installed. Both emit the same typed contract.
"""

from clothic.perception.base import PerceptionBackend
from clothic.perception.mock_backend import MockBackend

__all__ = ["PerceptionBackend", "MockBackend", "get_backend"]


def get_backend(name: str = "mock", **kwargs) -> PerceptionBackend:
    """Factory: return a perception backend by name.

    Falls back to the mock backend (with a warning) if a real backend's
    dependencies are unavailable, so the system always runs.
    """
    name = (name or "mock").lower()
    if name in ("mock", "sim", "simulator"):
        return MockBackend(**kwargs)
    if name in ("ultralytics", "yolo", "real"):
        try:
            from clothic.perception.ultralytics_backend import UltralyticsBackend

            return UltralyticsBackend(**kwargs)
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            return _fallback("Ultralytics", exc, kwargs)
    if name in ("full", "hybrid"):
        try:
            from clothic.perception.full_backend import FullBackend

            return FullBackend(**kwargs)
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            return _fallback("Full", exc, kwargs)
    raise ValueError(f"Unknown perception backend: {name!r}")


def _fallback(label: str, exc: Exception, kwargs: dict) -> PerceptionBackend:  # pragma: no cover
    import warnings

    warnings.warn(
        f"{label} backend unavailable ({exc}); falling back to MockBackend. "
        "Install with: pip install 'clothic[perception]'",
        RuntimeWarning,
        stacklevel=2,
    )
    # Mock backend doesn't accept real-backend kwargs; start it clean.
    return MockBackend()
