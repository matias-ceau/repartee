"""Fixtures for CLI tests: isolated git and XDG paths, a copy of the sample persona, a fake
model injected where the runtime builds its client, and the hash embedder in place of the
real embedding model. No network, no key."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from repartee.cli import main
from repartee.llm import FakeLLM
from repartee.llm.base import LLMUnavailable
from repartee.store.embed import HashEmbedder
from tests.home.helpers import GIT_ENV_TO_CLEAR, GuardedPopen
from tests.test_end_to_end import scripted_model

PERSONA = Path(__file__).resolve().parents[2] / "examples" / "persona"
NO_KEY = "No API key. Set ANTHROPIC_API_KEY or run `ant auth login`."


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate git and XDG paths; no key and the hash embedder unless a test says otherwise."""
    for name in GIT_ENV_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)
    env_dir = tmp_path / "_env"
    env_dir.mkdir()
    gitconfig = env_dir / "gitconfig"
    gitconfig.write_text("[init]\n\tdefaultBranch = main\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for var in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        path = env_dir / var.lower()
        path.mkdir()
        monkeypatch.setenv(var, str(path))
    monkeypatch.setattr(subprocess, "Popen", GuardedPopen)
    # As if the embedding model were already downloaded (no "downloading" note).
    (env_dir / "xdg_cache_home" / "repartee" / "models" / "model").mkdir(parents=True)

    def no_client(config, cache_dir=None):
        raise LLMUnavailable(NO_KEY)

    monkeypatch.setattr("repartee.runtime.get_client", no_client)
    monkeypatch.setattr("repartee.runtime.get_embedder",
                        lambda config, *, reasons=None: HashEmbedder() if config.enabled else None)
    return env_dir


@pytest.fixture(autouse=True)
def restore_logging():
    """The CLI configures the `repartee` logger; leave it as it was for other tests."""
    logger = logging.getLogger("repartee")
    handlers, level = list(logger.handlers), logger.level
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """A rule-based fake model, returned by every open_runtime of the test."""
    model = scripted_model()
    monkeypatch.setattr("repartee.runtime.get_client", lambda config, cache_dir=None: model)
    return model


@pytest.fixture
def persona(tmp_path: Path) -> Path:
    """A copy of the sample persona, in a folder named `persona` so the guard accepts it."""
    home = tmp_path / "persona"
    shutil.copytree(PERSONA, home)
    return home


Invoke = Callable[..., Result]


@pytest.fixture
def cli() -> Invoke:
    """Run the command line; `home` adds --home, `input` feeds stdin."""
    runner = CliRunner()

    def invoke(*args: str, home: Path | None = None, input: str | None = None) -> Result:
        argv = (["--home", str(home)] if home is not None else []) + list(args)
        return runner.invoke(main, argv, input=input, catch_exceptions=True)

    return invoke


@pytest.fixture
def tui_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, str]]:
    """Stand in for repartee.tui.app; records (runtime, mode) per run."""
    calls: list[tuple[object, str]] = []
    package = types.ModuleType("repartee.tui")
    app = types.ModuleType("repartee.tui.app")
    app.run = lambda rt, mode="chat": calls.append((rt, mode))
    package.app = app
    monkeypatch.setitem(sys.modules, "repartee.tui", package)
    monkeypatch.setitem(sys.modules, "repartee.tui.app", app)
    return calls


def one_line_error(result: Result) -> str:
    """The single stderr line of a refused command; fails on a traceback or several lines."""
    assert result.exit_code == 1, (result.exit_code, result.output, result.exception)
    assert "Traceback" not in result.output
    assert not isinstance(result.exception, Exception) or isinstance(result.exception, SystemExit)
    lines = [line for line in result.stderr.splitlines() if line.strip()]
    assert len(lines) == 1, result.stderr
    return lines[0]


def set_config(home: Path, extra: str) -> None:
    """Append YAML to the home's repartee.yaml (later keys win)."""
    path = home / "repartee.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "\n" + extra, encoding="utf-8")
