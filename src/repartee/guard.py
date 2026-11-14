"""Keep personal data and keys out of the public code repository.

Two safety nets:

- :func:`check_home_location` refuses a home that sits inside another git
  work tree (in practice: inside the code checkout), where its contents could
  be committed and pushed with the code.
- A scanner (:func:`scan_text`, :func:`scan_paths`, and ``python -m
  repartee.guard --staged`` used by ``scripts/pre-commit``) that flags chat
  exports, home data, credential files and API keys.

Rules, by name:

- ``chat-export-name``: ``_chat.txt``, ``WhatsApp Chat…``, ``Discussion WhatsApp…``,
  ``*.whatsapp.txt``.
- ``chat-export-lines``: three or more lines shaped like chat export messages
  (iOS ``[date, time] Name: text`` or Android ``date, time - Name: text``).
- ``home-config``: a ``repartee.yaml``.
- ``home-data``: a top-level home folder (``self/``, ``sources/``, ``interviews/``,
  ``conversations/``, ``drafts/``, ``derived/``) or a ``.claims/`` folder.
- ``credential-file``: ``.env``, ``.env.*`` (not ``.example``/``.sample``/``.template``),
  ``.pypirc``, ``.netrc``, SSH private key files.
- key rules (``anthropic-key``, ``openai-key``, ``pypi-token``, ``github-token``,
  ``aws-access-key``, ``google-api-key``, ``slack-token``, ``perplexity-key``,
  ``private-key``).

Paths under ``examples/persona/`` (fictional sample data) and ``tests/`` are
exempt from the chat, home-config and home-data rules, never from the key or
credential-file rules. Findings name the rule and line, never the secret.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath

from pydantic import BaseModel

from repartee.config import APP, xdg_config_home, xdg_data_home
from repartee.gitops import git_toplevel, remote_url


class UnsafeHome(Exception):
    """The home would sit where its contents could be committed with other code."""


class Finding(BaseModel):
    """One thing the scanner flagged. Never carries the flagged text itself."""

    path: str
    line: int | None
    rule: str
    detail: str

    def render(self) -> str:
        """``path:line: rule: detail`` for terminal output."""
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{where}: {self.rule}: {self.detail}"


# ------------------------------------------------------------------ location


def _is_persona(root: Path) -> bool:
    if root.parts[-2:] == ("examples", "persona"):
        return True
    # A copy of the sample persona made by a test or demo under the temp dir.
    temp = Path(tempfile.gettempdir()).resolve()
    return root.name == "persona" and temp in root.parents


def _enclosing_repo(root: Path) -> Path | None:
    """The other repository whose work tree contains root, or None when root is safe."""
    top = git_toplevel(root)
    if top is None:
        return None
    if top != root:
        return None if _is_persona(root) else top
    # root is a repository of its own; refuse it when nested in another work tree.
    if root.parent == root:
        return None
    return git_toplevel(root.parent)


def _suggested_location(avoid: Path) -> Path:
    default = (xdg_data_home() / APP).resolve()
    if default != avoid and _enclosing_repo(default) is None:
        return default
    return Path.home() / "repartee-home"


def check_home_location(root: Path) -> None:
    """Raise UnsafeHome when root is inside another git work tree (persona excepted)."""
    root = Path(root).expanduser().resolve()
    repo = _enclosing_repo(root)
    if repo is None:
        return
    url = remote_url(repo)
    remote = f" (remote: {url})" if url else ""
    suggestion = _suggested_location(root)
    user_config = xdg_config_home() / APP / "config.yaml"
    raise UnsafeHome(
        f"Refusing to use {root} as a Repartee home: it is inside the git repository at "
        f"{repo}{remote}. Anything Repartee keeps there (chat exports, journal, the "
        f"self-model) could be committed and pushed with that repository.\n"
        f"Put the home outside any other repository, for example {suggestion}, and point "
        f"Repartee at it with --home, $REPARTEE_HOME, or `home:` in {user_config}. "
        f"The home can be its own private git repository."
    )


# ------------------------------------------------------------------- scanner

_EXEMPT_PREFIXES = ("examples/persona/", "tests/")
_EXEMPTABLE = {"chat-export-name", "chat-export-lines", "home-config", "home-data"}
_HOME_FOLDERS = {"self", "sources", "interviews", "conversations", "drafts", "derived"}
_ENV_TEMPLATES = {"example", "sample", "template", "dist"}
_SSH_KEYS = {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"}
CHAT_LINE_THRESHOLD = 3

_INVISIBLE = "‎‏‪‫‬‭‮⁦⁧⁨⁩﻿"
_DATE = r"\d{1,4}[./-]\d{1,2}[./-]\d{1,4}\.?"
_TIME = r"\d{1,2}[:.]\d{2}(?:[:.]\d{2})?(?:\s*[AaPp]\.?\s?[Mm]\.?)?"
_AUTHOR = r"[^:\n\]]{1,80}?:(?:\s|$)"
_CHAT_LINE = re.compile(
    rf"^\[\s*{_DATE},?\s+{_TIME}\s*\]\s*{_AUTHOR}"  # iOS:     [12/03/2021, 14:05:33] Ana: hi
    rf"|^{_DATE},?\s+{_TIME}\s*[-–—]\s*{_AUTHOR}"  # Android: 12/03/2021, 14:05 - Ana: hi
)

_NB = r"(?<![A-Za-z0-9])"
_KEY_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("anthropic-key", re.compile(_NB + r"sk-ant-[A-Za-z0-9_\-]{10,}"), "an Anthropic API key"),
    ("openai-key", re.compile(_NB + r"sk-(?!ant-)[A-Za-z0-9_\-]{20,}"),
     "an API key starting with sk-"),
    ("pypi-token", re.compile(_NB + r"pypi-[A-Za-z0-9_\-]{32,}"), "a PyPI upload token"),
    ("github-token",
     re.compile(_NB + r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
     "a GitHub token"),
    ("aws-access-key", re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
     "an AWS access key id"),
    ("google-api-key", re.compile(_NB + r"AIza[0-9A-Za-z_\-]{35}"), "a Google API key"),
    ("slack-token", re.compile(_NB + r"xox[abposr]-[A-Za-z0-9-]{10,}"), "a Slack token"),
    ("perplexity-key", re.compile(_NB + r"pplx-[A-Za-z0-9]{20,}"), "a Perplexity API key"),
    ("private-key", re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY(?: BLOCK)?-----"),
     "a private key block"),
]
_KEY_ADVICE = "remove it and revoke the key; a pushed key is compromised"


def _normalize(path: str) -> str:
    text = path.replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _is_exempt(path: str) -> bool:
    return _normalize(path).startswith(_EXEMPT_PREFIXES)


def _name_findings(path: str) -> list[Finding]:
    norm = _normalize(path)
    parts = PurePosixPath(norm).parts
    if not parts:
        return []
    name = parts[-1]
    lower = name.casefold()
    found: list[Finding] = []

    def add(rule: str, detail: str) -> None:
        found.append(Finding(path=path, line=None, rule=rule, detail=detail))

    if (lower == "_chat.txt" or lower.startswith(("whatsapp chat", "discussion whatsapp"))
            or lower.endswith(".whatsapp.txt")):
        add("chat-export-name", "file name of a chat export; personal data belongs in the home")
    if (lower == ".env" or lower in {".pypirc", ".netrc"} or lower in _SSH_KEYS
            or (lower.startswith(".env.") and lower[5:] not in _ENV_TEMPLATES)):
        add("credential-file", "credentials file; keep it out of the repository")
    if lower == "repartee.yaml":
        add("home-config", "a home's repartee.yaml; the home lives outside this repository")
    if (len(parts) > 1 and parts[0] in _HOME_FOLDERS) or ".claims" in parts[:-1]:
        add("home-data", "path laid out like a Repartee home; the home lives elsewhere")
    return found


def _chat_findings(text: str, path: str) -> list[Finding]:
    first: int | None = None
    count = 0
    for number, line in enumerate(text.splitlines(), start=1):
        if _CHAT_LINE.match(line.lstrip(_INVISIBLE + " \t")):
            count += 1
            first = first or number
    if count < CHAT_LINE_THRESHOLD:
        return []
    return [Finding(path=path, line=first, rule="chat-export-lines",
                    detail=f"{count} lines shaped like chat export messages")]


def _key_findings(text: str, path: str) -> list[Finding]:
    found: list[Finding] = []
    seen: set[tuple[str, int]] = set()
    for rule, pattern, label in _KEY_RULES:
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            if (rule, line) in seen:
                continue
            seen.add((rule, line))
            found.append(Finding(path=path, line=line, rule=rule,
                                 detail=f"looks like {label}; {_KEY_ADVICE}"))
    found.sort(key=lambda f: (f.line or 0, f.rule))
    return found


def scan_text(text: str, path: str = "", *, binary: bool = False) -> list[Finding]:
    """Findings for one file's name and content. ``path`` is repository-relative when known."""
    exempt = _is_exempt(path)
    found = [f for f in _name_findings(path) if not (exempt and f.rule in _EXEMPTABLE)]
    if not exempt and not binary:
        found += _chat_findings(text, path)
    found += _key_findings(text, path)
    return found


def _decode(data: bytes) -> tuple[str, bool]:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace"), False
    binary = b"\x00" in data[:8192]
    return data.decode("utf-8", errors="replace"), binary


def scan_bytes(data: bytes, path: str) -> list[Finding]:
    """Findings for raw file content (decoded leniently; binary files get key rules only)."""
    text, binary = _decode(data)
    return scan_text(text, path, binary=binary)


_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".ruff_cache",
              ".pytest_cache", ".mypy_cache"}


def _iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for dirpath, dirnames, filenames in os.walk(path):
                dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
                for filename in sorted(filenames):
                    yield Path(dirpath) / filename
        else:
            yield path


def _display_path(path: Path, tops: dict[Path, Path | None]) -> str:
    parent = path.absolute().parent
    if parent not in tops:
        tops[parent] = git_toplevel(parent)
    top = tops[parent]
    if top is not None:
        try:
            return path.resolve().relative_to(top).as_posix()
        except ValueError:
            pass
    return str(path)


def scan_paths(paths: Iterable[Path]) -> list[Finding]:
    """Scan files and directories on disk (paths shown relative to their repository)."""
    found: list[Finding] = []
    tops: dict[Path, Path | None] = {}
    for path in _iter_files(paths):
        shown = _display_path(path, tops)
        try:
            data = path.read_bytes()
        except OSError as error:
            found.append(Finding(path=shown, line=None, rule="unreadable",
                                 detail=f"could not be read ({error.strerror or error})"))
            continue
        found += scan_bytes(data, shown)
    return found


# ------------------------------------------------------------- staged files


class GuardError(Exception):
    """The guard could not do its job (not a repository, git missing)."""


def _run_git(args: Sequence[str], cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    # The inherited environment is kept on purpose: during a commit, git points
    # GIT_INDEX_FILE at the index being committed.
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=False)
    except OSError as error:
        raise GuardError(f"cannot run git: {error}") from error


def scan_staged() -> list[Finding]:
    """Scan what is staged for commit in the repository of the current directory."""
    top = _run_git(["rev-parse", "--show-toplevel"])
    if top.returncode != 0:
        raise GuardError("not inside a git repository: "
                         + top.stderr.decode(errors="replace").strip())
    root = Path(top.stdout.decode().strip())
    listing = _run_git(["diff", "--cached", "--name-only", "--no-renames",
                        "--diff-filter=ACMRT", "-z"], cwd=root)
    if listing.returncode != 0:
        raise GuardError("git diff --cached failed: "
                         + listing.stderr.decode(errors="replace").strip())
    found: list[Finding] = []
    for raw in listing.stdout.split(b"\0"):
        if not raw:
            continue
        name = os.fsdecode(raw)
        blob = _run_git(["cat-file", "blob", f":0:{name}"], cwd=root)
        data = blob.stdout if blob.returncode == 0 else b""  # e.g. a submodule entry
        found += scan_bytes(data, name)
    return found


_BLOCKED_ADVICE = (
    "Personal data belongs in your private Repartee home, never in this repository.\n"
    "Unstage a file with `git restore --staged <path>`; revoke any key that was exposed.\n"
    "If a finding is wrong, check it by hand, then commit with --no-verify."
)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m repartee.guard [--staged | PATH...]``: 0 clean, 1 findings, 2 error."""
    parser = argparse.ArgumentParser(
        prog="python -m repartee.guard",
        description="Flag chat exports, home data, credential files and keys.",
    )
    parser.add_argument("--staged", action="store_true",
                        help="scan files staged for commit (the default without paths)")
    parser.add_argument("paths", nargs="*", type=Path, help="files or directories to scan")
    args = parser.parse_args(argv)

    try:
        findings = scan_paths(args.paths) if args.paths else []
        if args.staged or not args.paths:
            findings += scan_staged()
    except GuardError as error:
        print(f"repartee guard: {error}", file=sys.stderr)
        return 2

    if not findings:
        return 0
    for finding in findings:
        print(finding.render(), file=sys.stderr)
    plural = "s" if len(findings) != 1 else ""
    print(f"\nrepartee guard: {len(findings)} finding{plural}.\n{_BLOCKED_ADVICE}",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
