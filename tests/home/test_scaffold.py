"""Creating a home: layout, config, core page, .gitignore, repository."""

from __future__ import annotations

from pathlib import Path

import pytest

from repartee.config import HOME_CONFIG_NAME, HomeConfig
from repartee.guard import UnsafeHome
from repartee.home import Home
from repartee.scaffold import HomeExists, init_home
from tests.home.helpers import git, make_repo


def test_init_creates_layout_config_core_gitignore_and_repo(work: Path) -> None:
    root = work / "home"
    home = init_home(root, "Camille Martin")

    assert isinstance(home, Home)
    assert home.root == root.resolve()
    for directory in (home.sources_dir, home.interviews_dir, home.conversations_dir,
                      home.drafts_dir, home.self_dir, home.derived_dir, home.evals_dir):
        assert directory.is_dir()

    config = HomeConfig.load(root / HOME_CONFIG_NAME)
    assert config.owner.name == "Camille Martin"
    assert config.git.autocommit is True
    text = (root / HOME_CONFIG_NAME).read_text(encoding="utf-8")
    assert "# sources:" in text or "#   - type: whatsapp" in text  # commented examples

    core = (root / "self" / "core.md").read_text(encoding="utf-8")
    assert "## Pinned" in core
    start, end = "<!-- repartee:generated:start -->", "<!-- repartee:generated:end -->"
    assert start in core and end in core
    assert core.index(start) < core.index(end)
    assert core[core.index(start) + len(start):core.index(end)].strip() == ""

    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    for pattern in (".DS_Store", ".obsidian/workspace*", "*.swp"):
        assert pattern in gitignore.splitlines()

    assert (root / ".git").is_dir()
    # The new repository holds the scaffold in a first commit.
    assert git(root, "log", "--oneline").stdout.strip()
    assert git(root, "status", "--porcelain").stdout == ""


def test_init_owner_name_needing_quotes_round_trips(work: Path) -> None:
    init_home(work / "h", 'Zoë "Z" O\'Neil: #1')
    assert HomeConfig.load(work / "h" / HOME_CONFIG_NAME).owner.name == 'Zoë "Z" O\'Neil: #1'


def test_init_without_git(work: Path) -> None:
    init_home(work / "h", git=False)
    assert not (work / "h" / ".git").exists()
    assert (work / "h" / HOME_CONFIG_NAME).is_file()


def test_init_refuses_non_empty_non_home(work: Path) -> None:
    root = work / "old"
    root.mkdir()
    (root / "memory.db").write_text("old repartee data", encoding="utf-8")
    with pytest.raises(HomeExists) as info:
        init_home(root)
    message = str(info.value)
    assert str(root.resolve()) in message
    assert "force" in message
    assert not (root / HOME_CONFIG_NAME).exists()
    assert not (root / "self").exists()


def test_init_refuses_a_file(work: Path) -> None:
    target = work / "afile"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(HomeExists):
        init_home(target)


def test_init_accepts_fresh_clone_with_only_git_dir(work: Path) -> None:
    root = make_repo(work / "clone")
    home = init_home(root, "Me")
    assert home.exists()


def test_init_force(work: Path) -> None:
    root = work / "old"
    root.mkdir()
    (root / "memory.db").write_text("old repartee data", encoding="utf-8")
    home = init_home(root, "Me", force=True)
    assert home.exists()
    assert (root / "memory.db").read_text(encoding="utf-8") == "old repartee data"  # not migrated


def test_init_idempotent_on_existing_home(work: Path) -> None:
    root = work / "home"
    init_home(root, "Camille Martin")
    (root / HOME_CONFIG_NAME).write_text("owner:\n  name: Edited By Hand\n", encoding="utf-8")
    core = root / "self" / "core.md"
    core.write_text("my own core\n", encoding="utf-8")
    (root / ".gitignore").write_text("custom\n", encoding="utf-8")
    (root / "drafts").rmdir()

    init_home(root, "Someone Else")

    assert HomeConfig.load(root / HOME_CONFIG_NAME).owner.name == "Edited By Hand"
    assert core.read_text(encoding="utf-8") == "my own core\n"
    assert (root / ".gitignore").read_text(encoding="utf-8") == "custom\n"
    assert (root / "drafts").is_dir()


def test_init_checks_location_first(work: Path) -> None:
    repo = make_repo(work / "code")
    target = repo / "myhome"
    with pytest.raises(UnsafeHome):
        init_home(target)
    assert not target.exists()


def test_init_existing_repo_is_not_reinitialised(work: Path) -> None:
    root = make_repo(work / "home")
    (root / "README.md").write_text("private\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "first")
    init_home(root, force=True)
    log = git(root, "log", "--format=%s").stdout.splitlines()
    assert log == ["first"]  # init never commits in a repository it did not create
