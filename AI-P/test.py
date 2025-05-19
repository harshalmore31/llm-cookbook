import os
from mistralai import Mistral

api_key = "G4M4d2Ju73R4wmV1xsVsAea47cbDlERo"
client = Mistral(api_key=api_key)

model = "codestral-latest"
message = [{"role": "user", "content": "Write a function for fibonacci"}]
chat_response = client.chat.complete(
    model = model,
    messages = message
)
print(chat_response.choices[0].message.content)