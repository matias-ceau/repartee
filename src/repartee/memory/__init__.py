"""
This module manages different types of memory for the repartee library.

Features:
- Short term memory: Manages chat history using a large language model (LLM).
- Long term memory: Compresses and extracts important points from the chat history when it becomes too long.
- Semantic memory (RAG feature): Emulates semantic memory by retrieving and generating relevant information.

Usage:
Import this module to access memory management functionalities for chat history and semantic memory.
"""

from .short_term_memory import ShortTermMemory
from .semantic_memory import SemanticMemory
from .working_memory import WorkingMemory
from .episodic_memory import EpisodicMemory

# ---------- MCP host builder ----------
from fastmcp import Host

def build_host():
    """Return a Host exposing memory operations via MCP."""
    host = Host()

    from .short_term_memory import ShortTermMemory
    from .episodic_memory import EpisodicMemory

    stm = ShortTermMemory()
    epis = EpisodicMemory()

    @host.on("short.add_user")
    async def _su(msg):  # returns id ignored
        stm.add_user_message(msg)

    @host.on("short.get")
    async def _sg():
        return stm.get_messages_for_model()

    @host.on("short.add_assistant")
    async def _sa(msg):
        stm.add_assistant_message(msg)

    @host.on("short.add_system")
    async def _ss(msg):
        stm.add_system_message(msg)

    @host.on("episodic.add")
    async def _ea(role, content, conversation="default"):
        epis.add_message(role=role, content=content,
                         conversation_id=conversation)

    @host.on("episodic.search")
    async def _es(query, limit=3):
        return epis.search_similar(query, limit)

    return host
