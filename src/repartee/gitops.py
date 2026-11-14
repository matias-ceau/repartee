"""Local git for the home: find repositories and commit the home after a run.

Nothing here ever pushes. The home is committed locally; moving it between
machines is the owner's business.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from repartee.home import Home

log = logging.getLogger(__name__)

FALLBACK_IDENTITY = ("-c", "user.name=Repartee", "-c", "user.email=repartee@localhost")
TIMEOUT_SECONDS = 120

# Variables that would point git at another repository than the one asked for
# (they are set, for example, while a git hook runs).
_REPO_ENV = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_PREFIX",
)


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in _REPO_ENV}


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in ``cwd`` with an environment that cannot redirect it elsewhere."""
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        env=_clean_env(),
        timeout=TIMEOUT_SECONDS,
        check=False,
    )


def _existing_dir(path: Path) -> Path | None:
    current = Path(path).expanduser().absolute()
    while not current.is_dir():
        if current.parent == current:
            return None
        current = current.parent
    return current


def git_toplevel(path: Path) -> Path | None:
    """Top level of the git work tree containing path (or its nearest existing parent)."""
    start = _existing_dir(path)
    if start is None:
        return None
    try:
        result = _git(start, "rev-parse", "--show-toplevel")
    except (OSError, subprocess.SubprocessError):
        return None
    out = result.stdout.strip()
    if result.returncode != 0 or not out:
        return None
    return Path(out).resolve()


def is_git_repo(path: Path) -> bool:
    """True when path is inside a git work tree."""
    return git_toplevel(path) is not None


def remote_url(repo: Path) -> str | None:
    """URL of the repository's first remote (origin preferred), or None."""
    try:
        result = _git(repo, "remote", "get-url", "origin")
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        names = _git(repo, "remote").stdout.split()
        if names:
            result = _git(repo, "remote", "get-url", names[0])
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _has_identity(root: Path) -> bool:
    name = _git(root, "config", "--get", "user.name")
    email = _git(root, "config", "--get", "user.email")
    return bool(name.stdout.strip()) and bool(email.stdout.strip())


def autocommit(home: Home, message: str, *, enabled: bool = True) -> bool:
    """Commit everything in the home locally. True when a commit was made. Never pushes."""
    if not enabled:
        return False
    try:
        root = home.root.resolve()
        # Only a real home that is its own repository: never commit into an
        # enclosing repository (such as the code checkout).
        if not home.exists() or git_toplevel(root) != root:
            return False
        status = _git(root, "status", "--porcelain")
        if status.returncode != 0:
            log.warning("git status failed in %s: %s", root, status.stderr.strip())
            return False
        if not status.stdout.strip():
            return False
        added = _git(root, "add", "-A")
        if added.returncode != 0:
            log.warning("git add failed in %s: %s", root, added.stderr.strip())
            return False
        identity = () if _has_identity(root) else FALLBACK_IDENTITY
        commit = _git(root, *identity, "commit", "-q", "-m", message or "repartee: update")
        if commit.returncode != 0:
            log.warning("git commit failed in %s: %s", root, commit.stderr.strip())
            return False
        return True
    except Exception as error:  # noqa: BLE001 - autocommit must never break a run
        log.warning("autocommit of %s failed: %s", home.root, error)
        return False
