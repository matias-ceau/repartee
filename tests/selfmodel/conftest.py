"""Shared helpers for self-model tests. All people and facts here are invented."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repartee.selfmodel import SelfModel

ANA_KEY = "people/ana-ruiz"
C_LYON = "c-3fa91c20"
C_NURSE = "c-77be01aa"
C_TALK = "c-0d4e9a13"

ANA = """---
kind: person
title: Ana Ruiz
aliases: [Ana, Anita]
updated: 2026-11-14
---
# Ana Ruiz

## Who
- Friend from university; we met in Lyon in 2014. ^c-3fa91c20
- Works as a nurse, mostly night shifts. ^c-77be01aa

## How we talk
- We write in French, in short bursts, with a lot of teasing. ^c-0d4e9a13

## Notes
Anything I write here myself. The pipeline never touches it.
"""

NURSE_TEXT = "Works as a nurse, mostly night shifts."
NURSE_LINE = f"- {NURSE_TEXT} ^{C_NURSE}\n"

ANA_CLAIMS = {
    C_LYON: ("Who", "Friend from university; we met in Lyon in 2014."),
    C_NURSE: ("Who", "Works as a nurse, mostly night shifts."),
    C_TALK: ("How we talk", "We write in French, in short bursts, with a lot of teasing."),
}


def write_page(self_dir: Path, key: str, text: str | bytes) -> Path:
    path = self_dir / f"{key}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    return path


def read_page(self_dir: Path, key: str) -> str:
    return (self_dir / f"{key}.md").read_bytes().decode("utf-8")


def write_sidecar(self_dir: Path, key: str, claims: dict[str, tuple[str, str]], **extra) -> Path:
    """Sidecar with inferred pipeline claims: {id: (section, text)}."""
    path = self_dir / ".claims" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "version": 1,
        "claims": {
            cid: {
                "section": section,
                "text": text,
                "status": "inferred",
                "origin": "pipeline",
                "confidence": 0.5,
                "evidence": ["wa-1"],
            }
            for cid, (section, text) in claims.items()
        },
        "rejected": [],
    }
    body.update(extra)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def read_sidecar(self_dir: Path, key: str) -> dict:
    return json.loads((self_dir / ".claims" / f"{key}.json").read_text(encoding="utf-8"))


@pytest.fixture
def self_dir(tmp_path: Path) -> Path:
    d = tmp_path / "self"
    d.mkdir()
    return d


@pytest.fixture
def ana(self_dir: Path) -> Path:
    """A self/ folder holding Ana's page and its sidecar, all inferred."""
    write_page(self_dir, ANA_KEY, ANA)
    write_sidecar(self_dir, ANA_KEY, ANA_CLAIMS)
    return self_dir


def loaded(self_dir: Path) -> SelfModel:
    model = SelfModel(self_dir)
    model.load()
    return model


def evidence_known(*known: str):
    """An existing_evidence callable that knows only the given ids."""
    known_set = set(known)
    return lambda ids: {i for i in ids if i in known_set}
