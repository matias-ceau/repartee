"""Helpers for home, guard and git tests.

Every test here runs with git isolated from the developer's configuration, and
with a safety net: any git process that would change state inside the code
repository makes the test fail before git runs.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# git subcommands that only read. Anything else is refused inside the project.
READ_ONLY = {"rev-parse", "status", "diff", "show", "cat-file", "ls-files", "log", "--version"}

GIT_ENV_TO_CLEAR = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_AUTHOR_NAME",
    "GIT_AUTHOR_EMAIL",
    "GIT_COMMITTER_NAME",
    "GIT_COMMITTER_EMAIL",
    "REPARTEE_HOME",
)


def _inside_project(path: str | os.PathLike[str]) -> bool:
    resolved = Path(path).resolve()
    return resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents


def _git_target_and_subcommand(args: list[str], cwd: str | None) -> tuple[str, list[str]]:
    target = cwd or os.getcwd()
    rest: list[str] = []
    i = 1
    while i < len(args):
        arg = args[i]
        if arg == "-C" and i + 1 < len(args):
            target = os.path.join(target, args[i + 1])
            i += 2
            continue
        if arg == "-c" and i + 1 < len(args):
            i += 2
            continue
        rest = args[i:]
        break
    return target, rest


def _is_read_only(rest: list[str]) -> bool:
    if not rest:
        return True
    sub = rest[0]
    if sub in READ_ONLY:
        return True
    if sub == "config" and "--get" in rest:
        return True
    if sub == "remote" and (len(rest) == 1 or rest[1] in ("get-url", "-v")):
        return True
    return False


RealPopen = subprocess.Popen


class GuardedPopen(RealPopen):  # type: ignore[misc, valid-type]
    def __init__(self, args, *a, **kw):  # noqa: ANN001
        argv = [os.fspath(x) for x in ([args] if isinstance(args, (str, bytes)) else args)]
        if argv and os.path.basename(str(argv[0])) == "git":
            target, rest = _git_target_and_subcommand([str(x) for x in argv], kw.get("cwd"))
            env = kw.get("env") or os.environ
            redirects = [
                a.split("=", 1)[1] for a in argv if a.startswith(("--git-dir=", "--work-tree="))
            ] + [env[name] for name in ("GIT_DIR", "GIT_WORK_TREE") if env.get(name)]
            touches_project = _inside_project(target) or any(
                _inside_project(os.path.join(target, r)) for r in redirects
            )
            if touches_project and not _is_read_only(rest):
                raise RuntimeError(f"test tried to run a state-changing git command in the "
                                   f"code repository: {argv!r}")
        super().__init__(args, *a, **kw)


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run git in a temporary test repository. Refuses any path that is not a temp dir."""
    assert not _inside_project(repo), f"test git helper pointed at the code repository: {repo}"
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         *args],
        check=check,
        capture_output=True,
        text=True,
    )


def make_repo(path: Path) -> Path:
    """Create a git repository at path (under the test's temporary directory)."""
    assert not _inside_project(path), f"refusing to create a test repository at {path}"
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    return path
