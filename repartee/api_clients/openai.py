import openai
import os

class OpenAIAPIClient:
    """
    A client to interact with the OpenAI API.

    This class provides methods to send prompts to the OpenAI GPT-4 model and receive responses.

    Attributes
    ----------
    api_key : str
        The OpenAI API key used for authentication.

    Methods
    -------
    generate_text(prompt: str, model: str = "gpt-4", max_tokens: int = 100, temperature: float = 0.7) -> str
        Sends a prompt to the GPT-4 model and returns the generated response.
    """
    def __init__(self):
        """
        Initializes the OpenAIAPIClient with the API key from environment variables.
        
        Raises
        ------
        ValueError
            If the API key is not found in the environment variables.
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found in environment variables.")
        openai.api_key = self.api_key

    def generate_text(self, prompt: str, model: str = "gpt-4", max_tokens: int = 100, temperature: float = 0.7) -> str:
        """
        Sends a prompt to the GPT-4 model and returns the generated response.
        
        Parameters
        ----------
        prompt : str
            The prompt to send to the model.
        model : str, optional
            The model to use for text generation (default is "gpt-4").
        max_tokens : int, optional
            The maximum number of tokens to generate (default is 100).
        temperature : float, optional
            The sampling temperature to use (default is 0.7).

        Returns
        -------
        str
            The generated response from the model.

        Raises
        ------
        Exception
            If an error occurs during the API request.
        """
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            return f"An error occurred: {e}"

# Usage example
if __name__ == "__main__":
    client = OpenAIAPIClient()

    prompt = "Write a poem about the sea."
    text = client.generate_text(prompt)
    print("Generated Text:", text)
