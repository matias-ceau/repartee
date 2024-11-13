import sys

import anthropic

from ..models import AnthropicModel

client = anthropic.Anthropic()
if len(sys.argv) < 2:
    print("Usage: python -m repartee.models.anthropic_models.anthropic_models <prompt>")
    sys.exit(1)
else:
    prompt = sys.argv[1]
    model = AnthropicModel()
    response = model.generate_text(prompt)
