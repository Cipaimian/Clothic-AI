"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from clothic.config import load_profile


@pytest.fixture
def default_profile() -> dict:
    return load_profile("default")
