import sys

from ..models import OpenAIModel

prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello"
model = OpenAIModel()
response = model.send_message(prompt)
print(response)
