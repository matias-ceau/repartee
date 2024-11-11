from datetime import datetime


class EpisodicMemory:
    def __init__(self):
        self.memory = []

    def add(self, information):
        """
        Add episodic information to the memory with a timestamp.

        Args:
            information (str): The episodic information to store.
        """
        timestamp = datetime.now()
        self.memory.append((timestamp, information))

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
        Rank memories based on relevance. This is a placeholder for a more complex ranking algorithm.

        Returns:
            list: A list of tuples containing timestamps and episodic information, sorted by relevance.
        """
        # Placeholder: currently just returns memories sorted by timestamp
        return sorted(self.memory, key=lambda x: x[0], reverse=True)

    def __str__(self):
        return "\n".join(f"{timestamp}: {info}" for timestamp, info in self.memory)
