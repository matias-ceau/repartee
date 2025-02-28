"""
Integration module for OpenAI's GPT models.

Provides interfaces to communicate with OpenAI's API.
"""

from openai import OpenAI
from ..config import get_api_key


class OpenAIModel:
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        api_key = get_api_key("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key)
        
    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        max_tokens: int = 1024,
        conversation_history: list = None,
    ) -> str:
        messages = []
        
        # Add system message
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
            
        # Add current user message
        messages.append({"role": "user", "content": prompt})
        
        # Stream the response
        response_text = ""
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                response_text += content
                
        return response_text
