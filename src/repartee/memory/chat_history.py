import tiktoken


class ShortTermMemory:
    def __init__(self, history: None, max_tokens: int = 5000):
        self.history = [] if not history else history
        self.tokens = (
            0 if not history else len(tiktoken.tokenize("\n".join(self.history)))
        )

    def add(self, message):
        self.history.append(message)
        self._check_tokens()

    def _check_tokens(self):
        self.tokens = len(tiktoken.tokenize("\n".join(self.history)))
        if self.tokens > self.max_tokens:
            self.compress()

    def __str__(self):
        return "\n".join(self.history)

    def compress(self):
        """
        Compress the chat history by extracting important points.
        This method performs the following steps:
        1. Extracts key messages or points from the chat history.
        2. Removes redundant or less important information.
        3. Compiles the extracted points into a compressed format.
        Returns:
            None
        """
        # TODO: implement a compression method that first extracts important points from the chat history
        pass

class EpisodicMemory:
    pass

class LongTermMemory:
    pass