"""Everything a command needs, opened once: the home, the store, the self-model, the model client.

The CLI, the TUI and the evals all start from :func:`open_runtime`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from repartee.config import HomeConfig
from repartee.distill import Distiller
from repartee.guard import check_home_location
from repartee.home import Home
from repartee.llm import get_client
from repartee.llm.base import LLMClient, LLMUnavailable
from repartee.mirror import Mirror
from repartee.schema import Item
from repartee.selfmodel import SelfModel
from repartee.selfmodel.model import ReconcileReport
from repartee.store import Corpus
from repartee.store.embed import Embedder, get_embedder
from repartee.voice.cards import VoiceBuilder
from repartee.voice.drafter import Drafter, card_text
from repartee.voice.samples import pick_samples


class _VoiceLookup:
    """Adapts the voice code to what the mirror's get_voice tool asks for."""

    def __init__(self, rt: Runtime):
        self._rt = rt

    def registers(self) -> list[str]:
        return self._rt.model.voice_registers()

    def card(self, register_name: str) -> str | None:
        return card_text(self._rt.model, register_name, self._rt.config.owner.name)

    def samples(self, register_name: str, recipient: str | None, topic: str | None,
                k: int) -> list[Item]:
        return pick_samples(self._rt.corpus, register_name, recipient=recipient, topic=topic, k=k)


@dataclass
class Runtime:
    home: Home
    config: HomeConfig
    corpus: Corpus
    model: SelfModel
    llm: LLMClient | None
    #: Why there is no model client, as one line a person can act on.
    llm_error: str | None = None
    #: Why search is keyword-only, when it is.
    embedder_notes: list[str] = field(default_factory=list)
    #: What the owner changed by hand in the self-model since the last run.
    reconciled: ReconcileReport = field(default_factory=ReconcileReport)

    def require_llm(self) -> LLMClient:
        """The model client, or LLMUnavailable with the line that says how to get one."""
        if self.llm is None:
            raise LLMUnavailable(
                self.llm_error or "No model client. Set ANTHROPIC_API_KEY or run `ant auth login`."
            )
        return self.llm

    def distiller(self) -> Distiller:
        return Distiller(self.home, self.config, self.corpus, self.model, self.require_llm())

    def voice_builder(self) -> VoiceBuilder:
        """Style cards. Without a model client the cards hold statistics only."""
        return VoiceBuilder(self.home, self.corpus, self.model, self.llm)

    def drafter(self) -> Drafter:
        return Drafter(self.home, self.corpus, self.model, self.require_llm())

    def voice_lookup(self) -> _VoiceLookup:
        return _VoiceLookup(self)

    def mirror(self, mode: Literal["chat", "draft", "interview", "review"] = "chat") -> Mirror:
        mirror = Mirror(
            self.home, self.config, self.corpus, self.model, self.require_llm(),
            self.drafter(), voice=self.voice_lookup(),
        )
        mirror.set_mode(mode)
        return mirror

    def close(self) -> None:
        self.corpus.close()

    def __enter__(self) -> Runtime:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_runtime(
    home_path: str | Path | None = None,
    *,
    llm: LLMClient | None | Literal["auto"] = "auto",
    embedder: Embedder | None | Literal["auto"] = "auto",
) -> Runtime:
    """Open the home and everything on top of it.

    ``llm="auto"`` builds the configured client and leaves ``Runtime.llm`` as None
    (with ``llm_error`` set) when there are no credentials, so commands that need no
    model still work. ``embedder="auto"`` loads the configured embedding model, which
    downloads it on first use; pass None for commands that neither search nor ingest.

    Raises HomeNotFound when there is no home, UnsafeHome when it sits in another repository.
    """
    home = Home.open(home_path)
    check_home_location(home.root)
    config = home.load_config()
    home.ensure_dirs()

    notes: list[str] = []
    if embedder == "auto":
        embedder = get_embedder(config.embeddings, reasons=notes)
    corpus = Corpus(home.index_path, embedder)

    llm_error = None
    if llm == "auto":
        try:
            llm = get_client(config, home.cache_dir)
        except LLMUnavailable as error:
            llm, llm_error = None, str(error)

    model = SelfModel(home.self_dir)
    reconciled = model.load()
    if any(getattr(reconciled, name) for name in type(reconciled).model_fields):
        model.save()
        corpus.sync_claims(model.claims())

    return Runtime(home, config, corpus, model, llm, llm_error, notes, reconciled)
