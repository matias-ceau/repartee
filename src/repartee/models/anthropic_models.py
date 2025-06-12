"""
Integration module for Anthropic's language models.

Provides classes and methods to interact with Anthropic AI services.
"""

import anthropic

from ..config import ReparteeDefaults as Defaults


class AnthropicModel:
    @classmethod
    def list_models(cls): ...

    def __init__(self, model_name: str = ""):
        self.client = anthropic.Anthropic()
        self.model_name = model_name if model_name else Defaults.models.claude

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        conversation_history: list = None
    ) -> str:
        response_text = ""
        system_prompt = system_prompt if system_prompt else Defaults.system_prompt
        messages = []
        
        if conversation_history:
            messages = conversation_history
            
        messages.append({"role": "user", "content": prompt})
        
        with self.client.messages.stream(
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
            model=self.model_name,
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                response_text += text
            return response_text
