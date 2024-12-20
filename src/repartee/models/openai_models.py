"""
Integration module for OpenAI's GPT models.

Provides interfaces to communicate with OpenAI's API.
"""

import os

import openai


# TODO: fiw the wrong API call
class OpenAIModel:
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    def send_message(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        max_tokens: int = 1024,
    ) -> str:
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0]["message"]["content"]
