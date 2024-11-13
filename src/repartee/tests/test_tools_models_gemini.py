import sys

from ..models.google_models import GoogleModel

if len(sys.argv) < 2:
    prompt = "Hello"
else:
    prompt = sys.argv[1]
model = GoogleModel()
model.send(prompt)
