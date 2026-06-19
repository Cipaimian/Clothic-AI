"""Loading of policy profiles, thresholds, and pipeline config.

All behaviour-defining knobs live in editable JSON/YAML on disk so a campus
administrator can change rules and thresholds with no code change and no model
retraining. Profiles are versioned; the engine records which version judged
each decision.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Repo root = three levels up from this file (src/clothic/config.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES_DIR = REPO_ROOT / "campus_profiles"
CONFIGS_DIR = REPO_ROOT / "configs"
ONTOLOGY_DIR = REPO_ROOT / "ontology"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_profile(profile_id: str = "default", profiles_dir: Path | None = None) -> dict[str, Any]:
    """Load a campus policy profile by id (filename without extension)."""
    base = profiles_dir or PROFILES_DIR
    path = base / f"{profile_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Policy profile not found: {path}")
    return _read_json(path)


def list_profiles(profiles_dir: Path | None = None) -> list[str]:
    base = profiles_dir or PROFILES_DIR
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.json"))


@lru_cache(maxsize=1)
def load_thresholds() -> dict[str, Any]:
    path = CONFIGS_DIR / "thresholds.json"
    return _read_json(path) if path.exists() else {}


@lru_cache(maxsize=1)
def load_pipeline_config() -> dict[str, Any]:
    path = CONFIGS_DIR / "pipeline.yaml"
    return _read_yaml(path) if path.exists() else {}


# Keys in a backend block whose string values are on-disk artifacts and should
# be resolved relative to the repo root (so the API works from any cwd). Model
# refs like ``hf-hub:...`` or sentinels like ``sapiens`` are left untouched.
_PATH_KEYS = {"person_weights", "garment_weights", "pose_weights", "sapiens_checkpoint"}


def _resolve_artifact(value: Any) -> Any:
    """Resolve a relative artifact path against REPO_ROOT if it exists there."""
    if not isinstance(value, str) or not value:
        return value
    if "://" in value or value.startswith("hf-hub:"):
        return value  # remote / hub reference, not a local path
    p = Path(value)
    if p.is_absolute():
        return value
    candidate = REPO_ROOT / p
    return str(candidate) if candidate.exists() else value


def resolve_backend_config() -> tuple[str, dict[str, Any]]:
    """Resolve the active perception backend and its kwargs from pipeline.yaml.

    Single source of truth for "which models run". The chosen backend name comes
    from ``CLOTHIC_BACKEND`` if set, else the ``backend:`` key in pipeline.yaml,
    else ``mock``. Its kwargs come from the matching block under ``backends:``,
    with artifact paths resolved against the repo root. The mock backend takes no
    kwargs, so an empty block is fine.
    """
    cfg = load_pipeline_config()
    backend = os.environ.get("CLOTHIC_BACKEND") or cfg.get("backend", "mock")
    block = dict((cfg.get("backends") or {}).get(backend, {}) or {})
    kwargs = {k: (_resolve_artifact(v) if k in _PATH_KEYS else v) for k, v in block.items()}
    return backend, kwargs


@lru_cache(maxsize=1)
def load_taxonomy() -> dict[str, Any]:
    path = ONTOLOGY_DIR / "taxonomy.yaml"
    return _read_yaml(path) if path.exists() else {}


def load_calibration() -> dict[str, float]:
    """Optional per-head temperatures from configs/calibration.json.

    Absent file => empty dict => identity calibration (a no-op). Produced by
    ``tools/fit_calibration.py`` from a labeled validation set.
    """
    path = CONFIGS_DIR / "calibration.json"
    return _read_json(path).get("temperatures", {}) if path.exists() else {}
