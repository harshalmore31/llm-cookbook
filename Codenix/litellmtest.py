from litellm import completion
import os

os.environ["OPENAI_API_KEY"] = ""

messages = [{ "content": "Hello, how are you?","role": "user"}]
response = completion(model="openai/gpt-4o", messages=messages)


print(response)