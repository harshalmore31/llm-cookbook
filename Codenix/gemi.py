from litellm import completion
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
os.environ['GEMINI_API_KEY'] = api_key
user_input = input("user_input : ")
response = completion(
    model="gemini/gemini-pro", 
    messages=[
        {"role": "user", "content": user_input},
         {"role": "system", "content": "You are an Assistant"} # System message for context.
    ]
)


print("Gemini : "+response.choices[0].message.content)