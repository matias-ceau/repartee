"""Create a home: folders, ``repartee.yaml``, ``self/core.md``, ``.gitignore``, a repository."""

from __future__ import annotations

import datetime as dt
import logging
import subprocess
from pathlib import Path

import yaml

from repartee.gitops import autocommit, is_git_repo
from repartee.guard import check_home_location
from repartee.home import Home

log = logging.getLogger(__name__)

# Entries that do not make a directory "already used": an empty clone of the
# owner's private repository, or Finder litter.
_IGNORABLE_ENTRIES = {".git", ".DS_Store"}

_CONFIG_TEMPLATE = """\
# Repartee home. Everything in this folder is private: keep it out of public repositories.
{owner}
# Where your material is. Paths are relative to this folder, absolute, or ~/...
sources: []
# sources:
#   - type: whatsapp            # WhatsApp exports (_chat.txt, folders or zips)
#     path: sources/whatsapp
#   - type: vault               # an Obsidian vault, may live outside the home
#     path: ~/Documents/vault
#     exclude: ["Templates/**"]
#     include: ["People/**", "Books/**"]  # optional: only these paths
#     kinds:                    # glob -> kind; frontmatter `kind:` wins
#       "Journal/**": journal
#       "Health/**": medical
#       "Essays/**": essay
#   - type: writing             # essays and other texts
#     path: sources/writing
#   - type: tastes              # music and book lists (CSV, JSON, markdown)
#     path: sources/tastes
#   - type: assistant           # Claude.ai or ChatGPT exports (zip, folder, conversations.json)
#     path: sources/assistant   # only your own turns count; the assistant's are context

# models: {{bulk: claude-opus-5-5, smart: claude-opus-5-5, chat: claude-opus-5-5}}
# effort: {{bulk: low, smart: high, chat: medium}}
# embeddings: {{enabled: true}}

git:
  autocommit: true              # commit this folder locally after each run; never pushes
"""

_CORE_TEMPLATE = """\
---
kind: core
title: Core
aliases: []
updated: {today}
---
# Core

## Pinned

## Generated
<!-- repartee:generated:start -->
<!-- repartee:generated:end -->
"""

_GITIGNORE = """\
# OS and editor litter
.DS_Store
Thumbs.db
*.swp
*.swo
*~
.#*
# Obsidian per-device state
.obsidian/workspace*
"""


class HomeExists(Exception):
    """The target directory is in use by something that is not a Repartee home."""


def _owner_block(owner_name: str) -> str:
    return yaml.safe_dump({"owner": {"name": owner_name, "aliases": []}},
                          sort_keys=False, allow_unicode=True).rstrip("\n")


def _write_if_missing(path: Path, text: str) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def _check_target(root: Path, home: Home, force: bool) -> None:
    if root.exists() and not root.is_dir():
        raise HomeExists(f"{root} exists and is not a directory.")
    if not root.is_dir() or home.exists() or force:
        return
    entries = sorted(p.name for p in root.iterdir() if p.name not in _IGNORABLE_ENTRIES)
    if entries:
        shown = ", ".join(entries[:5]) + (", ..." if len(entries) > 5 else "")
        raise HomeExists(
            f"{root} already has files ({shown}) but no {home.config_path.name}, so it is "
            f"not a Repartee home (it may hold data from the old Repartee, which is not "
            f"migrated). Choose an empty directory, or use --force to add a home here."
        )


def _git_init(root: Path) -> bool:
    try:
        result = subprocess.run(["git", "-C", str(root), "init", "-q"],
                                capture_output=True, text=True, check=False)
    except OSError as error:
        log.warning("git init skipped in %s: %s", root, error)
        return False
    if result.returncode != 0:
        log.warning("git init failed in %s: %s", root, result.stderr.strip())
        return False
    return True


def init_home(path: Path, owner_name: str = "Me", *, force: bool = False,
              git: bool = True) -> Home:
    """Create a home at path, or add the missing pieces to an existing one."""
    root = Path(path).expanduser().resolve()
    check_home_location(root)
    home = Home(root)
    _check_target(root, home, force)

    root.mkdir(parents=True, exist_ok=True)
    home.ensure_dirs()
    _write_if_missing(home.config_path, _CONFIG_TEMPLATE.format(owner=_owner_block(owner_name)))
    _write_if_missing(home.self_dir / "core.md",
                      _CORE_TEMPLATE.format(today=dt.date.today().isoformat()))
    _write_if_missing(root / ".gitignore", _GITIGNORE)

    if git and not is_git_repo(root) and _git_init(root):
        autocommit(home, "Create Repartee home")
    return home
