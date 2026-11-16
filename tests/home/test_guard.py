"""Home location check, personal-data scanner, staged scan and pre-commit hook.

All sample secrets below are fake and assembled from pieces at run time, so this
file itself never contains a string the scanner would flag.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from repartee import guard
from repartee.guard import Finding, UnsafeHome, check_home_location, scan_paths, scan_text
from tests.home.helpers import PROJECT_ROOT, git, make_repo

# --------------------------------------------------------------- fake secrets

FAKE_SECRETS = {
    "anthropic-key": "sk-" + "ant-" + "api03-" + "Zq8" * 12,
    "openai-key": "sk-" + "proj-" + "Q7w" * 10,
    "pypi-token": "py" + "pi-" + "AgEIcHlwaS5vcmc" + "Kx9" * 15,
    "github-token": "gh" + "p_" + "R2d" * 12,
    "aws-access-key": "AK" + "IA" + "QWERTYUIOPASDFGH",
    "private-key": "-----BEGIN " + "OPENSSH PRIVATE" + " KEY-----",
}

IOS_LINES = [
    "[12/03/2021, 14:05:33] Ana Ruiz: salut",
    "[12/03/2021, 14:06:01] Cam: ça va ?",
    "‎[12/03/2021, 14:07:12] Ana Ruiz: oui et toi",
]
ANDROID_LINES = [
    "3/12/21, 2:05 PM - Ana: hi",
    "3/12/21, 2:06 PM - Cam: hey",
    "3/25/21, 11:40 AM - Ana: see you",
]
ANDROID_24H_LINES = [
    "12.03.21, 14:05 - Ana: hallo",
    "12.03.21, 14:06 - Cam: hi",
    "12.03.21, 14:07 - Ana: tschüss",
]


def rules(findings: list[Finding]) -> list[str]:
    return [f.rule for f in findings]


def assert_no_secret_shown(findings: list[Finding], secret: str) -> None:
    for finding in findings:
        dumped = finding.model_dump_json()
        assert secret not in dumped
        assert secret[:12] not in dumped


# ------------------------------------------------------------------- location


def test_home_inside_other_repo_is_unsafe(work: Path) -> None:
    repo = make_repo(work / "code")
    (repo / "sub").mkdir()
    with pytest.raises(UnsafeHome) as info:
        check_home_location(repo / "sub")
    message = str(info.value)
    assert str(repo.resolve()) in message  # which repository it would leak into
    assert "REPARTEE_HOME" in message or "--home" in message  # how to point elsewhere
    assert str(Path(os.environ["XDG_DATA_HOME"]) / "repartee") in message  # where instead


def test_home_not_yet_created_inside_repo_is_unsafe(work: Path) -> None:
    repo = make_repo(work / "code")
    with pytest.raises(UnsafeHome):
        check_home_location(repo / "deep" / "new" / "home")


def test_unsafe_message_names_remote(work: Path) -> None:
    repo = make_repo(work / "code")
    git(repo, "remote", "add", "origin", "https://example.invalid/someone/code.git")
    with pytest.raises(UnsafeHome, match="example.invalid/someone/code.git"):
        check_home_location(repo / "home")


def test_home_as_its_own_repo_is_safe(work: Path) -> None:
    home = make_repo(work / "home")
    check_home_location(home)
    plain = work / "plain"
    plain.mkdir()
    check_home_location(plain)
    check_home_location(work / "not-created-yet")


def test_home_repo_nested_inside_other_repo_is_unsafe(work: Path) -> None:
    outer = make_repo(work / "code")
    inner = make_repo(outer / "home")
    with pytest.raises(UnsafeHome, match=str(outer.resolve())):
        check_home_location(inner)


def test_persona_path_exempt(work: Path) -> None:
    repo = make_repo(work / "code")
    persona = repo / "examples" / "persona"
    persona.mkdir(parents=True)
    check_home_location(persona)
    with pytest.raises(UnsafeHome):
        check_home_location(repo / "examples" / "other")
    with pytest.raises(UnsafeHome):
        check_home_location(persona / "nested")


def test_persona_copy_under_temp_dir_exempt(work: Path) -> None:
    repo = make_repo(work / "sandbox")
    copy = repo / "persona"
    copy.mkdir()
    check_home_location(copy)


def test_real_code_checkout(tmp_path: Path) -> None:
    probe = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip("code checkout is not a git repository")
    with pytest.raises(UnsafeHome, match=str(PROJECT_ROOT)):
        check_home_location(PROJECT_ROOT / "home")
    check_home_location(PROJECT_ROOT / "examples" / "persona")


# -------------------------------------------------------------------- scanner


def test_scan_flags_chat_export_shapes_names_and_keys() -> None:
    for lines in (IOS_LINES, ANDROID_LINES, ANDROID_24H_LINES):
        found = scan_text("intro\n" + "\n".join(lines) + "\n", "notes/log.txt")
        assert rules(found) == ["chat-export-lines"], lines
        assert found[0].line == 2
        for line in lines:
            assert line.split(": ", 1)[1] not in found[0].detail

    for name in ("_chat.txt", "dump/_chat.txt", "WhatsApp Chat with Ana.txt",
                 "x/WhatsApp Chat - Group.zip", "ana.whatsapp.txt", "dump/conversations.json"):
        assert "chat-export-name" in rules(scan_text("", name)), name

    for name in (".env", "deploy/.env.production", ".pypirc", "x/.netrc"):
        assert "credential-file" in rules(scan_text("", name)), name

    assert "home-config" in rules(scan_text("owner: {}\n", "repartee.yaml"))
    assert "home-config" in rules(scan_text("owner: {}\n", "somewhere/repartee.yaml"))

    for name in ("self/core.md", "sources/whatsapp/a.txt", "derived/obs.jsonl",
                 "interviews/2025.md", "x/.claims/people/ana.json"):
        assert "home-data" in rules(scan_text("", name)), name


@pytest.mark.parametrize("rule", sorted(FAKE_SECRETS))
def test_scan_flags_each_key_rule_without_showing_it(rule: str) -> None:
    secret = FAKE_SECRETS[rule]
    text = f"line one\nconfig = '{secret}'\n"
    found = scan_text(text, "src/settings.py")
    assert rules(found) == [rule]
    assert found[0].line == 2
    assert found[0].path == "src/settings.py"
    assert_no_secret_shown(found, secret)


def test_scan_two_chat_lines_are_not_enough() -> None:
    assert scan_text("\n".join(IOS_LINES[:2]), "notes.md") == []


def test_scan_clean_text_and_lookalikes() -> None:
    text = (
        "Run the task-runner with --disk-cache.\n"
        "The skeleton key idea; sk-short.\n"
        "Meeting at 14:05 - Ana said hi\n"
        "from 2021 to 2022: nothing\n"
    )
    assert scan_text(text, "docs/readme.md") == []
    assert scan_text("", ".env.example") == []
    assert scan_text("", "src/repartee/home.py") == []


def test_scan_exempts_persona_and_tests_but_not_keys() -> None:
    chat = "\n".join(IOS_LINES + ANDROID_LINES)
    for path in ("examples/persona/sources/whatsapp/_chat.txt",
                 "examples/persona/sources/whatsapp/WhatsApp Chat with Ana.txt",
                 "examples/persona/repartee.yaml",
                 "examples/persona/self/core.md",
                 "tests/ingest/fixtures/_chat.txt",
                 "tests/home/repartee.yaml"):
        assert scan_text(chat, path) == [], path

    secret = FAKE_SECRETS["anthropic-key"]
    for path in ("examples/persona/sources/vault/note.md", "tests/test_x.py"):
        found = scan_text(f"key: {secret}\n", path)
        assert rules(found) == ["anthropic-key"], path
        assert_no_secret_shown(found, secret)

    # Credential files are never exempt.
    assert rules(scan_text("", "tests/.env")) == ["credential-file"]
    # A similar-looking path elsewhere is not exempt.
    assert "chat-export-lines" in rules(scan_text(chat, "src/tests_data/chat.md"))
    assert "chat-export-name" in rules(scan_text("", "old/examples/persona/_chat.txt"))


def test_this_test_file_is_clean() -> None:
    here = Path(__file__)
    findings = scan_text(here.read_text(encoding="utf-8"), "tests/home/test_guard.py")
    assert findings == []


def test_scan_paths_reads_files_and_directories(work: Path) -> None:
    secret = FAKE_SECRETS["github-token"]
    (work / "a").mkdir()
    (work / "a" / "settings.toml").write_text(f"token = '{secret}'\n", encoding="utf-8")
    (work / "a" / "_chat.txt").write_text("\n".join(IOS_LINES), encoding="utf-8")
    (work / "clean.md").write_text("nothing here\n", encoding="utf-8")
    (work / "blob.bin").write_bytes(b"\x00\x01" + secret.encode() + b"\xff\xfe")

    found = scan_paths([work])
    by_rule = sorted(rules(found))
    assert by_rule == ["chat-export-lines", "chat-export-name", "github-token", "github-token"]
    assert_no_secret_shown(found, secret)


def test_scan_paths_repo_relative_exemption(work: Path) -> None:
    repo = make_repo(work / "code")
    fixture = repo / "tests" / "data" / "_chat.txt"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("\n".join(IOS_LINES), encoding="utf-8")
    assert scan_paths([fixture]) == []
    stray = repo / "notes" / "_chat.txt"
    stray.parent.mkdir()
    stray.write_text("\n".join(IOS_LINES), encoding="utf-8")
    assert "chat-export-name" in rules(scan_paths([stray]))


# ------------------------------------------------------------- staged and hook


def test_guard_staged_returns_1_on_finding(
    work: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = make_repo(work / "code")
    monkeypatch.chdir(repo)
    (repo / "ok.py").write_text("print('hi')\n", encoding="utf-8")
    git(repo, "add", "ok.py")
    assert guard.main(["--staged"]) == 0

    secret = FAKE_SECRETS["pypi-token"]
    (repo / "upload.cfg").write_text(f"password = {secret}\n", encoding="utf-8")
    (repo / "export").mkdir()
    (repo / "export" / "_chat.txt").write_text("\n".join(ANDROID_LINES), encoding="utf-8")
    git(repo, "add", "-A")
    capsys.readouterr()
    assert guard.main(["--staged"]) == 1
    out = capsys.readouterr()
    text = out.out + out.err
    assert "upload.cfg:1" in text and "pypi-token" in text
    assert "export/_chat.txt" in text
    assert secret not in text and secret[:12] not in text


def test_guard_staged_scans_staged_content_not_working_tree(
    work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(work / "code")
    monkeypatch.chdir(repo)
    target = repo / "notes.md"
    target.write_text("\n".join(IOS_LINES) + "\n", encoding="utf-8")
    git(repo, "add", "notes.md")
    target.write_text("cleaned up in the working tree only\n", encoding="utf-8")
    assert guard.main(["--staged"]) == 1


def test_guard_paths_cli(work: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (work / "clean.md").write_text("fine\n", encoding="utf-8")
    assert guard.main([str(work / "clean.md")]) == 0
    (work / ".env").write_text("A=1\n", encoding="utf-8")
    assert guard.main([str(work)]) == 1
    assert "credential-file" in capsys.readouterr().err


def test_guard_staged_outside_repo_fails_closed(
    work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(work)
    assert guard.main(["--staged"]) == 2


def test_pre_commit_script_is_executable_and_runs(work: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "pre-commit"
    assert script.is_file()
    assert script.stat().st_mode & stat.S_IXUSR
    assert script.read_text(encoding="utf-8").startswith("#!/bin/sh")

    repo = make_repo(work / "code")
    hook = repo / ".git" / "hooks" / "pre-commit"
    shutil.copy2(script, hook)
    env = {**os.environ, "REPARTEE_PYTHON": sys.executable}

    def commit(message: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=t@example.invalid",
             "commit", "-q", "-m", message],
            capture_output=True, text=True, env=env,
        )

    (repo / "ok.py").write_text("print('hi')\n", encoding="utf-8")
    git(repo, "add", "ok.py")
    clean = commit("clean")
    assert clean.returncode == 0, clean.stderr

    (repo / "WhatsApp Chat with Ana.txt").write_text("\n".join(IOS_LINES), encoding="utf-8")
    git(repo, "add", "-A")
    blocked = commit("personal")
    assert blocked.returncode != 0
    assert "chat-export" in blocked.stderr
    assert git(repo, "log", "--format=%s").stdout.splitlines() == ["clean"]
