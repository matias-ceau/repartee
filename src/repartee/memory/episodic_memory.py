"""
Episodic memory with vector storage for long-term conversation history.

This module provides functionality to store, retrieve, and search through 
past conversations using vector embeddings.
"""

import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from openai import OpenAI

from ..config import get_api_key


class EpisodicMemory:
    """
    Long-term conversation memory using vector embeddings for semantic search.
    
    Uses SQLite and vector embeddings to store and retrieve conversation history,
    providing semantic search capabilities over past interactions.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the episodic memory system.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses a default path.
        """
        # Set up database path
        if db_path is None:
            # Default to user's home directory under .repartee
            home = Path.home()
            db_dir = home / ".repartee" / "memory"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "episodic_memory.db")
            
        self.db_path = db_path
        self._init_database()
        
        # Basic in-memory storage for compatibility with old code
        self.memory = []
        
        # Initialize embedding API client
        api_key = get_api_key("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Embeddings require an API key.")
        self.client = OpenAI(api_key=api_key)
        
    def _init_database(self):
        """Set up the SQLite database schema if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            title TEXT,
            metadata TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            embedding BLOB
        )
        ''')
        
        # Create indexes for faster querying
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_conversation_id ON messages(conversation_id)
        ''')
        
        conn.commit()
        conn.close()
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding vector for the given text.
        
        Args:
            text: The text to embed
            
        Returns:
            Numpy array containing the embedding vector
        """
        from ..config import config
        response = self.client.embeddings.create(
            input=text,
            model=config.embeddings["model"]
        )
        return np.array(response.data[0].embedding)
    
    # Legacy methods for backwards compatibility
    def add(self, information):
        """
        Add episodic information to the memory with a timestamp.

        Args:
            information (str): The episodic information to store.
        """
        timestamp = datetime.now()
        self.memory.append((timestamp, information))
        
        # Also add to vector store
        self.add_message("user", information)

    def retrieve(self, limit=None):
        """
        Retrieve episodic information from the memory, optionally limiting the number of entries.

        Args:
            limit (int, optional): The maximum number of entries to retrieve. If None, retrieve all entries.

        Returns:
            list: A list of tuples containing timestamps and episodic information.
        """
        if limit is not None:
            return self.memory[-limit:]
        return self.memory

    def remove(self, timestamp):
        """
        Remove episodic information from the memory by timestamp.

        Args:
            timestamp (datetime): The timestamp of the information to remove.
        """
        self.memory = [entry for entry in self.memory if entry[0] != timestamp]

    def rank_memories(self):
        """
        Rank memories based on relevance.

        Returns:
            list: A list of tuples containing timestamps and episodic information, sorted by relevance.
        """
        # Return memories sorted by timestamp
        return sorted(self.memory, key=lambda x: x[0], reverse=True)
        
    # New vector-based methods
    def add_message(self, 
                   role: str, 
                   content: str, 
                   conversation_id: str = "default",
                   timestamp: str = None):
        """
        Add a single message to the episodic memory.
        
        Args:
            role: The role of the message sender (user, assistant, system)
            content: The content of the message
            conversation_id: Identifier for the conversation this message belongs to
            timestamp: Optional timestamp, defaults to current time
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if conversation exists, create if not
        cursor.execute(
            "SELECT COUNT(*) FROM conversations WHERE conversation_id = ?", 
            (conversation_id,)
        )
        
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                '''
                INSERT INTO conversations (conversation_id, timestamp, title) 
                VALUES (?, ?, ?)
                ''', 
                (conversation_id, timestamp, f"Conversation {conversation_id}")
            )
        
        # Generate embedding
        try:
            embedding = self._get_embedding(content)
            embedding_bytes = embedding.tobytes()
        except Exception as e:
            # If embedding fails, store without it
            print(f"Warning: Failed to generate embedding: {e}")
            embedding_bytes = None
        
        # Store message
        cursor.execute(
            '''
            INSERT INTO messages (conversation_id, role, content, timestamp, embedding) 
            VALUES (?, ?, ?, ?, ?)
            ''', 
            (conversation_id, role, content, timestamp, embedding_bytes)
        )
        
        conn.commit()
        conn.close()
        
    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for messages similar to the query across all conversations.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of message dictionaries with similarity scores
        """
        query_embedding = self._get_embedding(query)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all messages with their embeddings
        cursor.execute(
            "SELECT id, conversation_id, role, content, timestamp, embedding FROM messages"
        )
        messages = cursor.fetchall()
        conn.close()
        
        results = []
        for msg in messages:
            msg_id, conv_id, role, content, timestamp, embedding_bytes = msg
            
            if embedding_bytes:
                # Convert embedding from binary to numpy array
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                
                message_data = {
                    "id": msg_id,
                    "conversation_id": conv_id,
                    "role": role,
                    "content": content,
                    "timestamp": timestamp,
                    "similarity": float(similarity)
                }
                
                results.append(message_data)
        
        # Sort by similarity score in descending order and return top results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]
    
    def get_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve a full conversation by its ID.
        
        Args:
            conversation_id: The ID of the conversation to retrieve
            
        Returns:
            List of message dictionaries in chronological order
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT id, role, content, timestamp 
            FROM messages 
            WHERE conversation_id = ? 
            ORDER BY timestamp
            """, 
            (conversation_id,)
        )
        
        messages = []
        for msg_id, role, content, timestamp in cursor.fetchall():
            messages.append({
                "id": msg_id,
                "role": role,
                "content": content,
                "timestamp": timestamp
            })
            
        conn.close()
        return messages
    
    def list_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent conversations with their metadata.
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation dictionaries with metadata
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT conversation_id, timestamp, title, metadata 
            FROM conversations 
            ORDER BY timestamp DESC 
            LIMIT ?
            """, 
            (limit,)
        )
        
        conversations = []
        for conv_id, timestamp, title, metadata_json in cursor.fetchall():
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            # Get message count for this conversation
            cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", 
                (conv_id,)
            )
            message_count = cursor.fetchone()[0]
            
            conversations.append({
                "conversation_id": conv_id,
                "timestamp": timestamp,
                "title": title,
                "message_count": message_count,
                "metadata": metadata
            })
            
        conn.close()
        return conversations
        
    def __str__(self):
        """String representation of memory for debugging."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM conversations")
        conv_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM messages")
        msg_count = cursor.fetchone()[0]
        
        conn.close()
        
        return f"EpisodicMemory: {conv_count} conversations, {msg_count} messages"
