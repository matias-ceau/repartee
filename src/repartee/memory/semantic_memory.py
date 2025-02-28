"""
Semantic memory module for Repartee.

Provides knowledge graph functionality for storing and retrieving conceptual 
information about the user and their knowledge domain.
"""

import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional

import numpy as np
from openai import OpenAI

from ..config import get_api_key


class KnowledgeGraph:
    """
    Knowledge graph implementation for semantic memory.
    
    Stores concepts, relations, and metadata in a graph structure that can be
    queried for relationships between concepts.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the knowledge graph.
        
        Args:
            db_path: Path to store the graph database. If None, uses a default path.
        """
        # Set up database path
        if db_path is None:
            # Default to user's home directory under .repartee
            home = Path.home()
            db_dir = home / ".repartee" / "memory"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "knowledge_graph.db")
            
        self.db_path = db_path
        self._init_database()
        
        # Initialize embedding API client for semantic similarity
        api_key = get_api_key("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Semantic search requires an API key.")
        self.client = OpenAI(api_key=api_key)
        
    def _init_database(self):
        """Set up the SQLite database schema if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            type TEXT,
            metadata TEXT,
            embedding BLOB,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            relation TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (source_id) REFERENCES nodes (id),
            FOREIGN KEY (target_id) REFERENCES nodes (id),
            UNIQUE (source_id, target_id, relation)
        )
        ''')
        
        # Create indexes for faster querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_node_name ON nodes(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edge_source ON edges(source_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edge_target ON edges(target_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edge_relation ON edges(relation)')
        
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
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return np.array(response.data[0].embedding)
    
    def add_concept(self, 
                   name: str, 
                   concept_type: str = None, 
                   metadata: Dict[str, Any] = None) -> int:
        """
        Add a concept node to the knowledge graph.
        
        Args:
            name: The name of the concept
            concept_type: Optional type classification for the concept
            metadata: Optional additional information about the concept
            
        Returns:
            ID of the created node
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        
        # Generate embedding for semantic similarity
        try:
            embedding = self._get_embedding(name)
            embedding_bytes = embedding.tobytes()
        except Exception as e:
            print(f"Warning: Failed to generate embedding: {e}")
            embedding_bytes = None
        
        # Check if node already exists
        cursor.execute("SELECT id FROM nodes WHERE name = ?", (name,))
        existing = cursor.fetchone()
        
        if existing:
            node_id = existing[0]
            # Update existing node
            cursor.execute(
                """
                UPDATE nodes 
                SET type = ?, metadata = ?, embedding = ?, updated_at = ? 
                WHERE id = ?
                """,
                (concept_type, metadata_json, embedding_bytes, current_time, node_id)
            )
        else:
            # Create new node
            cursor.execute(
                """
                INSERT INTO nodes (name, type, metadata, embedding, created_at, updated_at) 
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (name, concept_type, metadata_json, embedding_bytes, current_time, current_time)
            )
            node_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return node_id
    
    def add_relation(self,
                    source: str,
                    relation: str,
                    target: str,
                    weight: float = 1.0,
                    metadata: Dict[str, Any] = None) -> Tuple[int, int, int]:
        """
        Add a relation between two concepts.
        
        Args:
            source: Source concept name
            relation: Type of relationship
            target: Target concept name
            weight: Strength of relationship (0.0-1.0)
            metadata: Optional additional information about the relationship
            
        Returns:
            Tuple of (source_id, target_id, edge_id)
        """
        # Ensure both concepts exist
        source_id = self.add_concept(source)
        target_id = self.add_concept(target)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        
        # Check if relation already exists
        cursor.execute(
            """
            SELECT id FROM edges 
            WHERE source_id = ? AND target_id = ? AND relation = ?
            """, 
            (source_id, target_id, relation)
        )
        existing = cursor.fetchone()
        
        if existing:
            edge_id = existing[0]
            # Update existing relation
            cursor.execute(
                """
                UPDATE edges 
                SET weight = ?, metadata = ?, updated_at = ? 
                WHERE id = ?
                """,
                (weight, metadata_json, current_time, edge_id)
            )
        else:
            # Create new relation
            cursor.execute(
                """
                INSERT INTO edges 
                (source_id, target_id, relation, weight, metadata, created_at, updated_at) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (source_id, target_id, relation, weight, metadata_json, current_time, current_time)
            )
            edge_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return (source_id, target_id, edge_id)
    
    def get_related_concepts(self, 
                            concept: str, 
                            relation: str = None,
                            direction: str = "outgoing") -> List[Dict[str, Any]]:
        """
        Get concepts related to the given concept.
        
        Args:
            concept: The concept to find relations for
            relation: Optional specific relation type to filter by
            direction: Whether to get "incoming", "outgoing", or "both" relations
            
        Returns:
            List of related concepts with their relation information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get the concept ID
        cursor.execute("SELECT id FROM nodes WHERE name = ?", (concept,))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return []
            
        concept_id = result[0]
        related = []
        
        # Outgoing relations (concept → other)
        if direction in ["outgoing", "both"]:
            query = """
            SELECT e.relation, e.weight, n.id, n.name, n.type, n.metadata 
            FROM edges e 
            JOIN nodes n ON e.target_id = n.id 
            WHERE e.source_id = ?
            """
            params = [concept_id]
            
            if relation:
                query += " AND e.relation = ?"
                params.append(relation)
                
            cursor.execute(query, params)
            
            for rel, weight, node_id, name, node_type, metadata_json in cursor.fetchall():
                metadata = json.loads(metadata_json) if metadata_json else {}
                related.append({
                    "concept": name,
                    "concept_id": node_id,
                    "type": node_type,
                    "relation": rel,
                    "direction": "outgoing",
                    "weight": weight,
                    "metadata": metadata
                })
        
        # Incoming relations (other → concept)
        if direction in ["incoming", "both"]:
            query = """
            SELECT e.relation, e.weight, n.id, n.name, n.type, n.metadata 
            FROM edges e 
            JOIN nodes n ON e.source_id = n.id 
            WHERE e.target_id = ?
            """
            params = [concept_id]
            
            if relation:
                query += " AND e.relation = ?"
                params.append(relation)
                
            cursor.execute(query, params)
            
            for rel, weight, node_id, name, node_type, metadata_json in cursor.fetchall():
                metadata = json.loads(metadata_json) if metadata_json else {}
                related.append({
                    "concept": name,
                    "concept_id": node_id,
                    "type": node_type,
                    "relation": rel,
                    "direction": "incoming",
                    "weight": weight,
                    "metadata": metadata
                })
        
        conn.close()
        return related
    
    def search_similar_concepts(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for concepts semantically similar to the query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of concepts with similarity scores
        """
        query_embedding = self._get_embedding(query)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all nodes with their embeddings
        cursor.execute("SELECT id, name, type, metadata, embedding FROM nodes")
        nodes = cursor.fetchall()
        conn.close()
        
        results = []
        for node in nodes:
            node_id, name, node_type, metadata_json, embedding_bytes = node
            
            if embedding_bytes:
                # Convert embedding from binary to numpy array
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                results.append({
                    "concept_id": node_id,
                    "concept": name,
                    "type": node_type,
                    "similarity": float(similarity),
                    "metadata": metadata
                })
        
        # Sort by similarity score in descending order and return top results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]
    
    def import_from_obsidian(self, folder_path: str, 
                            import_links: bool = True, 
                            import_tags: bool = True) -> Tuple[int, int]:
        """
        Import concepts and relations from Obsidian notes.
        
        Args:
            folder_path: Path to Obsidian vault folder
            import_links: Whether to import links between notes as relations
            import_tags: Whether to import tags as concept types
            
        Returns:
            Tuple of (concepts_added, relations_added)
        """
        concepts_added = 0
        relations_added = 0
        
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Invalid Obsidian folder path: {folder_path}")
        
        # Process markdown files recursively
        for md_file in folder.glob("**/*.md"):
            # Skip template files in Obsidian
            if ".obsidian/templates" in str(md_file):
                continue
                
            # Get note name from filename
            note_name = md_file.stem
            
            # Extract note content and process
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Add the note as a concept
                note_metadata = {
                    "source": "obsidian",
                    "file_path": str(md_file),
                    "created": datetime.fromtimestamp(
                        md_file.stat().st_ctime
                    ).isoformat(),
                    "modified": datetime.fromtimestamp(
                        md_file.stat().st_mtime
                    ).isoformat()
                }
                
                # Find tags if enabled
                note_type = None
                if import_tags:
                    tags = self._extract_tags(content)
                    if tags:
                        note_type = tags[0]  # Use first tag as type
                        note_metadata["tags"] = tags
                        
                # Add the note as a concept
                concept_id = self.add_concept(
                    name=note_name,
                    concept_type=note_type,
                    metadata=note_metadata
                )
                concepts_added += 1
                
                # Process links if enabled
                if import_links:
                    linked_notes = self._extract_links(content)
                    for linked_note in linked_notes:
                        # Add relation to each linked note
                        _, _, edge_id = self.add_relation(
                            source=note_name,
                            relation="links_to",
                            target=linked_note,
                            metadata={"source": "obsidian_link"}
                        )
                        relations_added += 1
                        
            except Exception as e:
                print(f"Error processing {md_file}: {e}")
                continue
                
        return (concepts_added, relations_added)
    
    def _extract_links(self, content: str) -> Set[str]:
        """Extract wiki-style links from Obsidian note content."""
        import re
        links = set()
        
        # Match [[Link]] and [[Link|Alias]] patterns
        link_pattern = r'\[\[(.*?)(?:\|.*?)?\]\]'
        matches = re.findall(link_pattern, content)
        
        for match in matches:
            # Remove any section references with #
            clean_link = match.split('#')[0].strip()
            if clean_link:
                links.add(clean_link)
                
        return links
    
    def _extract_tags(self, content: str) -> List[str]:
        """Extract tags from Obsidian note content."""
        import re
        tags = []
        
        # Match #tag patterns, but not inside code blocks
        # This is a simplified approach
        tag_pattern = r'(?<!\S)#([a-zA-Z0-9_\-/]+)'
        matches = re.findall(tag_pattern, content)
        
        for match in matches:
            if match:
                tags.append(match)
                
        return tags


class SemanticMemory:
    """
    Semantic memory for storing and retrieving conceptual knowledge.
    
    Uses a knowledge graph to represent concepts and their relationships,
    providing semantic search and reasoning capabilities.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the semantic memory system.
        
        Args:
            db_path: Optional custom path for the database
        """
        self.knowledge_graph = KnowledgeGraph(db_path=db_path)
        self.memory = {}  # Legacy storage for backward compatibility
        
    # Legacy methods for backward compatibility
    def add(self, key, information):
        """
        Add semantic information to the memory.

        Args:
            key (str): The key associated with the information.
            information (str): The semantic information to store.
        """
        self.memory[key] = information
        
        # Also add to knowledge graph
        self.add_concept(key, "memory_key", {"content": information})

    def retrieve(self, key):
        """
        Retrieve semantic information from the memory.

        Args:
            key (str): The key associated with the information to retrieve.

        Returns:
            str: The retrieved semantic information, or None if the key does not exist.
        """
        return self.memory.get(key, None)

    def remove(self, key):
        """
        Remove semantic information from the memory.

        Args:
            key (str): The key associated with the information to remove.
        """
        if key in self.memory:
            del self.memory[key]
            
    # New knowledge graph methods    
    def add_concept(self, 
                   name: str, 
                   concept_type: str = None, 
                   metadata: Dict[str, Any] = None) -> int:
        """Add a concept to semantic memory."""
        return self.knowledge_graph.add_concept(name, concept_type, metadata)
        
    def add_relation(self,
                    source: str,
                    relation: str,
                    target: str,
                    weight: float = 1.0,
                    metadata: Dict[str, Any] = None) -> Tuple[int, int, int]:
        """Add a relation between concepts."""
        return self.knowledge_graph.add_relation(source, relation, target, weight, metadata)
        
    def get_related_concepts(self, 
                            concept: str, 
                            relation: str = None,
                            direction: str = "both") -> List[Dict[str, Any]]:
        """Get concepts related to the specified concept."""
        return self.knowledge_graph.get_related_concepts(concept, relation, direction)
        
    def search_similar_concepts(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for concepts similar to the query."""
        return self.knowledge_graph.search_similar_concepts(query, limit)
        
    def import_from_obsidian(self, folder_path: str) -> Tuple[int, int]:
        """Import concepts and relations from Obsidian notes."""
        return self.knowledge_graph.import_from_obsidian(folder_path)
        
    def __str__(self):
        """String representation of semantic memory."""
        legacy_count = len(self.memory.items())
        return f"SemanticMemory: {legacy_count} legacy items, {str(self.knowledge_graph)}"
