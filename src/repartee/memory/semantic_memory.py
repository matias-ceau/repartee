class SemanticMemory:
    """
    Semantic memory implementation based on the neurological concept.

    This class manages the storage and retrieval of semantic information.
    """

    def __init__(self):
        self.memory = {}

    def add(self, key, information):
        """
        Add semantic information to the memory.

        Args:
            key (str): The key associated with the information.
            information (str): The semantic information to store.
        """
        self.memory[key] = information

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

    def __str__(self):
        return "\n".join(f"{key}: {info}" for key, info in self.memory.items())
