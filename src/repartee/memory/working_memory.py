class WorkingMemory:
    """
    Working memory implementation for handling dynamic information that the model needs to have in context.

    This class manages the temporary storage and retrieval of information that is actively being used by the model.
    """

    def __init__(self, max_size: int = 100):
        self.memory = []
        self.max_size = max_size

    def add(self, item):
        """
        Add a new item to the working memory.

        If the memory exceeds the maximum size, the oldest item is removed.
        """
        self.memory.append(item)
        if len(self.memory) > self.max_size:
            self.memory.pop(0)

    def get_all(self):
        """
        Retrieve all items currently in the working memory.
        """
        return self.memory

    def clear(self):
        """
        Clear all items from the working memory.
        """
        self.memory = []

    def __str__(self):
        return "\n".join(str(item) for item in self.memory)
