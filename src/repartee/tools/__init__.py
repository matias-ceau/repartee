"""
This module provides tools for:

- Coding assistance
- Web search capabilities
- File system access for models
- Didactic functions based on Retrieval-Augmented Generation (RAG) from documentation and manuals.
"""

class Tool:
    """
    The Tool class represents a generic tool with basic functionality.
    """

    def __init__(self, name):
        self.name = name

    def use(self):
        print(f"Using {self.name}")