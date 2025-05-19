from litellm import completion
from dotenv import load_dotenv
import json

load_dotenv()

def get_current_weather(location, unit="fahrenheit"):
    if location == "Boston, MA":
        return f"The weather is {12 if unit == 'fahrenheit' else -11}°{'F' if unit == 'fahrenheit' else 'C'}"
    return "Location not found"

messages = [{"role": "user", "content": "What is the weather like in Boston?"}]

functions = [{
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
}]

# First call to identify function
response = completion(
    model="gpt-4o", 
    messages=messages, 
    functions=functions
    )

function_call = response["choices"][0]["message"].get("function_call")

if function_call:
    # Access attributes properly based on the object type
    function_name = function_call.name if hasattr(function_call, "name") else function_call["name"]
    function_args_str = function_call.arguments if hasattr(function_call, "arguments") else function_call["arguments"]
    
    # Parse the arguments
    function_args = json.loads(function_args_str)
    result = get_current_weather(**function_args)
    
    # Update messages with function call and result
    messages.extend([
        {"role": "assistant", "content": None, "function_call": {
            "name": function_name,
            "arguments": function_args_str
        }},
        {"role": "function", "name": function_name, "content": result}
    ])
    
    # Second call to process function result
    response = completion(model="gpt-4o", messages=messages, functions=functions)
    print(response["choices"][0]["message"]["content"])