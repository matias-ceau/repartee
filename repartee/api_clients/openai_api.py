import os
import openai

# Initialize the OpenAI client with the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

openai.api_key = api_key

def get_openai_response(prompt, model="gpt-4"):
    """
    Send a prompt to the OpenAI API and return the response.

    :param prompt: The prompt to send to the API.
    :param model: The model to use for the API call (default: "gpt-4").
    :return: The response from the API.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return response.choices[0].message['content']
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    test_prompt = "Say this is a test"
    response = get_openai_response(test_prompt)
    if response:
        print(f"OpenAI response: {response}")
    else:
        print("Failed to get a response from OpenAI.")