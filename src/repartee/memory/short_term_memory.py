import json
import tiktoken
from datetime import datetime


class ShortTermMemory:
    """
    Short-term memory management for conversation context.

    Maintains the most recent conversation history within a token limit,
    providing efficient access to the current conversation context.
    """

    def __init__(self, max_tokens: int = 5000):
        self.max_tokens = max_tokens
        self.conversation_history = []
        self.token_count = 0
        self.encoding = tiktoken.get_encoding(
            "cl100k_base"
        )  # Compatible with most models

    def add_message(self, role: str, message_text: str):
        """
        Add a message to the conversation history.
        Args:
            role (str): Who is sending the message ('user', 'assistant', or 'system').
            message_text (str): The message content.
        """
        if role not in {"user", "assistant", "system"}:
            raise ValueError("Role must be 'user', 'assistant', or 'system'")
        message = {
            "role": "user",
            "content": message_text,
            "timestamp": datetime.now().isoformat(),
        }
        # System messages at the beginning, others at the end
        if role == "system":
            self.conversation_history.insert(0, message)
        else:
            self.conversation_history.append(message)
        self._update_token_count()
        return message

    def _update_token_count(self):
        """Update the token count and trim history if needed."""
        # Approximate token count from conversation history
        all_text = json.dumps(self.conversation_history)
        self.token_count = len(self.encoding.encode(all_text))

        # If over max tokens, remove oldest messages (except system)
        while self.token_count > self.max_tokens and len(self.conversation_history) > 1:
            # Keep system messages at the beginning
            for i, msg in enumerate(self.conversation_history):
                if (
                    i > 0 and msg.get("role") != "system"
                ):  # Skip first message if it's system
                    self.conversation_history.pop(i)
                    break

            # Recalculate token count
            all_text = json.dumps(self.conversation_history)
            self.token_count = len(self.encoding.encode(all_text))

    def get_messages_for_model(self):
        """Return the conversation history formatted for model API calls."""
        model_messages = []
        for msg in self.conversation_history:
            # Create a clean copy without timestamps for the API
            model_messages.append({"role": msg["role"], "content": msg["content"]})
        return model_messages

    def clear(self):
        """Clear all conversation history except system messages."""
        system_messages = [
            msg for msg in self.conversation_history if msg["role"] == "system"
        ]
        self.conversation_history = system_messages
        self._update_token_count()

    def __str__(self):
        """Format conversation history as a readable string."""
        output = []
        for message in self.conversation_history:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            timestamp = message.get("timestamp", "")
            output.append(f"[{timestamp}] {role.capitalize()}: {content}")
        return "\n".join(output)
