"""Shared fixtures for ingest tests. Fixtures are small and invented."""

from __future__ import annotations

import pytest

from repartee.config import OwnerConfig


@pytest.fixture
def owner() -> OwnerConfig:
    return OwnerConfig(name="Camille Martin", aliases=["Camille", "Cam"])
