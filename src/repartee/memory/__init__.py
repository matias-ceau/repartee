"""
This module manages different types of memory for the repartee library.

Features:
- Short term memory: Manages chat history using a large language model (LLM).
- Long term memory: Compresses and extracts important points from the chat history when it becomes too long.
- Semantic memory (RAG feature): Emulates semantic memory by retrieving and generating relevant information.

Usage:
Import this module to access memory management functionalities for chat history and semantic memory.
"""
from .chat_history import ShortTermMemory, LongTermMemory, EpisodicMemory
from .rag import SemanticMemory