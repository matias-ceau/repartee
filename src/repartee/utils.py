from dataclasses import dataclass

import openai


def oneshot_call(model_name: str, prompt: str, max_tokens: int = 1024) -> str:
    """
    Sends a single message to the specified model and returns the response.
    """
    response = openai.Completion.create(
        model=model_name, prompt=prompt, max_tokens=max_tokens
    )
    return response.choices[0].text.strip()
