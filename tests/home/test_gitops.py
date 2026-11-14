"""Local auto-commit of the home."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from repartee import gitops
from repartee.gitops import autocommit, git_toplevel, is_git_repo
from repartee.home import Home
from tests.home.helpers import PROJECT_ROOT, git, make_repo


def test_is_git_repo_and_toplevel(work: Path) -> None:
    assert not is_git_repo(work)
    assert git_toplevel(work) is None
    repo = make_repo(work / "repo")
    (repo / "sub").mkdir()
    assert is_git_repo(repo / "sub")
    assert git_toplevel(repo / "sub") == repo.resolve()
    # A path that does not exist yet resolves through its nearest existing parent.
    assert git_toplevel(repo / "not" / "yet") == repo.resolve()


def _home_repo(path: Path) -> Home:
    root = make_repo(path)
    (root / "repartee.yaml").write_text("owner:\n  name: Test Owner\n", encoding="utf-8")
    return Home(root.resolve())


def _commits(repo: Path) -> list[str]:
    result = git(repo, "log", "--format=%s", check=False)
    return result.stdout.splitlines() if result.returncode == 0 else []


def test_autocommit_commits_changes_once_and_never_pushes(
    work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = work / "remote.git"
    remote.mkdir()
    git(remote, "init", "-q", "--bare")
    home = _home_repo(work / "home")
    root = home.root
    git(root, "remote", "add", "origin", str(remote))
    (root / "self").mkdir()
    (root / "self" / "core.md").write_text("hello\n", encoding="utf-8")

    seen: list[list[str]] = []
    real_run = subprocess.run

    def recording_run(args, *a, **kw):  # noqa: ANN001, ANN202
        seen.append([str(x) for x in args])
        return real_run(args, *a, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(gitops.subprocess, "run", recording_run)
        assert autocommit(home, "pipeline run") is True
        assert _commits(root) == ["pipeline run"]
        assert autocommit(home, "again") is False  # nothing changed
        assert _commits(root) == ["pipeline run"]

        (root / "self" / "core.md").write_text("hello again\n", encoding="utf-8")
        assert autocommit(home, "second") is True
        assert _commits(root) == ["second", "pipeline run"]

    assert not any("push" in args for args in seen)
    assert git(remote, "for-each-ref").stdout == ""  # remote untouched


def test_autocommit_uses_fallback_identity_only_when_none_configured(work: Path) -> None:
    home = _home_repo(work / "home")
    root = home.root
    (root / "a.md").write_text("a\n", encoding="utf-8")
    assert autocommit(home, "one")
    assert git(root, "log", "-1", "--format=%an <%ae>").stdout.strip() == (
        "Repartee <repartee@localhost>"
    )

    git(root, "config", "user.name", "Camille")
    git(root, "config", "user.email", "camille@example.invalid")
    (root / "b.md").write_text("b\n", encoding="utf-8")
    assert autocommit(home, "two")
    assert git(root, "log", "-1", "--format=%an <%ae>").stdout.strip() == (
        "Camille <camille@example.invalid>"
    )


def test_autocommit_noop_when_disabled_or_not_repo(work: Path) -> None:
    plain = work / "plain"
    plain.mkdir()
    (plain / "a.md").write_text("a\n", encoding="utf-8")
    assert autocommit(Home(plain.resolve()), "x") is False
    assert not (plain / ".git").exists()

    home = _home_repo(work / "home")
    (home.root / "a.md").write_text("a\n", encoding="utf-8")
    assert autocommit(home, "x", enabled=False) is False
    assert _commits(home.root) == []

    # A repository without repartee.yaml is not a home: never committed.
    other = make_repo(work / "other")
    (other / "a.md").write_text("a\n", encoding="utf-8")
    assert autocommit(Home(other.resolve()), "x") is False
    assert _commits(other) == []

    assert autocommit(Home(work / "missing"), "x") is False


def test_autocommit_refuses_home_that_is_a_subfolder_of_a_repo(work: Path) -> None:
    repo = make_repo(work / "code")
    (repo / "README.md").write_text("code\n", encoding="utf-8")
    sub = repo / "home"
    sub.mkdir()
    (sub / "journal.md").write_text("private\n", encoding="utf-8")
    (sub / "repartee.yaml").write_text("owner:\n  name: Test Owner\n", encoding="utf-8")
    assert autocommit(Home(sub.resolve()), "x") is False
    assert _commits(repo) == []
    assert git(repo, "diff", "--cached", "--name-only").stdout == ""


def test_autocommit_never_raises(work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = _home_repo(work / "home")
    root = home.root
    (root / "a.md").write_text("a\n", encoding="utf-8")

    def missing_git(*a, **kw):  # noqa: ANN002, ANN003, ANN202
        raise FileNotFoundError("git")

    with monkeypatch.context() as patch:
        patch.setattr(gitops.subprocess, "run", missing_git)
        assert autocommit(home, "x") is False

    # A failing commit (a pre-commit hook that rejects) also returns False.
    hook = root / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    assert autocommit(home, "x") is False


def test_autocommit_on_code_repository_does_nothing() -> None:
    before = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"], capture_output=True, text=True
    )
    if before.returncode != 0:
        pytest.skip("code checkout is not a git repository")
    assert autocommit(Home(PROJECT_ROOT), "must not happen") is False


def test_safety_net_blocks_state_changing_git_in_code_repository() -> None:
    with pytest.raises(RuntimeError, match="state-changing"):
        subprocess.run(["git", "-C", str(PROJECT_ROOT), "add", "--dry-run", "-A"])
    with pytest.raises(RuntimeError, match="state-changing"):
        subprocess.run(["git", f"--git-dir={PROJECT_ROOT / '.git'}", "add", "--dry-run", "-A"])
    with pytest.raises(AssertionError):
        make_repo(PROJECT_ROOT / "tmp-test-repo")
    assert not (PROJECT_ROOT / "tmp-test-repo").exists()
