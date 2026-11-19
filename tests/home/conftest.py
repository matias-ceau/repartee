"""Fixtures for home, guard and git tests: isolated git config and XDG paths, plus a
safety net that fails any test whose git process would change the code repository."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.home.helpers import GIT_ENV_TO_CLEAR, GuardedPopen


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate git and XDG paths, and refuse state-changing git in the code repository."""
    for name in GIT_ENV_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)
    env_dir = tmp_path / "_env"
    env_dir.mkdir()
    gitconfig = env_dir / "gitconfig"
    gitconfig.write_text("[init]\n\tdefaultBranch = main\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    for var in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        path = env_dir / var.lower()
        path.mkdir()
        monkeypatch.setenv(var, str(path))
    monkeypatch.setattr(subprocess, "Popen", GuardedPopen)
    return env_dir


@pytest.fixture
def work(tmp_path: Path) -> Path:
    """An empty working directory inside the test's temporary directory."""
    path = tmp_path / "work"
    path.mkdir()
    return path
