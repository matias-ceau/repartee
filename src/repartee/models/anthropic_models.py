"""
Integration module for Anthropic's language models.

Provides classes and methods to interact with Anthropic AI services.
"""

import anthropic


class AnthropicModel:
    @classmethod
    def list_models(cls):
        client = anthropic.Anthropic()
        models = client.models.list()
        return [model["id"] for model in models]

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022"):
        self.client = anthropic.Anthropic()
        self.model_name = model_name

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        max_tokens: int = 1024,
    ) -> str:
        response_text = ""
        with self.client.messages.stream(
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=self.model_name,
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                response_text += text
            return response_text


if __name__ == "__main__":
    import sys

    client = anthropic.Anthropic()
    if len(sys.argv) < 2:
        print(
            "Usage: python -m repartee.models.anthropic_models.anthropic_models <prompt>"
        )
        sys.exit(1)
    else:
        prompt = sys.argv[1]
        model = AnthropicModel()
        response = model.generate_text(prompt)

