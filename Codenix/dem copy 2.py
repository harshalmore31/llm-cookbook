from litellm import models_by_provider

providers = [
    "openai",
    "anthropic",
    "cohere",
    "gemini",
    "mistral",
    "groq",
    "nvidia_nim",
    "huggingface",
    "perplexity"
]

filtered_models = {}

for provider in providers:
    filtered_models[provider] = models_by_provider.get(provider, [])

print(filtered_models)