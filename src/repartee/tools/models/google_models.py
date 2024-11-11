"""
Integration module for Google's language models.

Facilitates interaction with Google's AI platforms.
"""

import google.generativeai as genai
import os
from typing import Dict, List, Union

genai.configure(api_key=os.environ["GEMINI_API_KEY"])


class GoogleModel:
    def __init__(
        self, model_name: str = "gemini-1.5-flash", history: List[Dict[str, str]] = None
    ):
        # self.history = history if history else []
        self.model = genai.GenerativeModel(model_name)
        self.chat = self.model.start_chat()

    def send(self, message: str, stream: bool = True) -> Union[str, List[str]]:
        response = self.chat.send_message(message, stream=stream)
        if stream:
            for chunk in response:
                print(chunk.text)
                print("_" * 80)
            return response.text_stream
        else:
            print(response.text)
            return response.text


# TODO: make coherent with other models (priority = low)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        prompt = "Hello"
    else:
        prompt = sys.argv[1]
    model = GoogleModel()
    model.send(prompt)
