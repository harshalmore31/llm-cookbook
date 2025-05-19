from litellm import completion
from dotenv import load_dotenv
import inspect
import json

# Load environment variables (if any)
load_dotenv()

### 📌 Define functions (Tools) ###
def get_current_weather(location, unit="fahrenheit"):
    """Returns the current weather for a given location."""
    if location == "Boston, MA":
        return f"The weather is {12 if unit == 'fahrenheit' else -11}°{'F' if unit == 'fahrenheit' else 'C'}"
    return "Location not found"

def get_time_zone(city):
    """Returns the time zone of a city."""
    time_zones = {"New York": "EST", "Los Angeles": "PST", "London": "GMT"}
    return time_zones.get(city, "Time zone not found")

### 📌 Function to generate tool schema dynamically ###
def generate_function_schemas(functions):
    """Automatically generates JSON schemas for function calling."""
    function_schemas = []
    
    for func in functions:
        sig = inspect.signature(func)
        params = {
            param: {"type": "string"} for param in sig.parameters
        }
        function_schemas.append({
            "name": func.__name__,
            "description": func.__doc__ or "No description provided.",
            "parameters": {
                "type": "object",
                "properties": params,
                "required": list(sig.parameters.keys())
            }
        })
    
    return function_schemas

### 📌 Function to handle dynamic function calling ###
def execute_llm_query(user_query):
    """Handles LLM interaction, detects function calls, and executes functions dynamically."""
    
    # Step 1: Auto-generate function schemas
    available_functions = [get_current_weather, get_time_zone]
    function_map = {func.__name__: func for func in available_functions}  # Map name to function
    function_schemas = generate_function_schemas(available_functions)

    # Step 2: Send user query to LLM
    messages = [{"role": "user", "content": user_query}]
    response = completion(model="gpt-4o", messages=messages, functions=function_schemas)

    # Step 3: Extract function call(s)
    function_calls = response["choices"][0]["message"].get("function_call")
    if not function_calls:
        return response["choices"][0]["message"]["content"]

    if not isinstance(function_calls, list):  
        function_calls = [function_calls]  # Convert to list if it's a single function call

    # Step 4: Execute functions dynamically
    results = []
    for function_call in function_calls:
        function_name = function_call["name"]
        function_args = json.loads(function_call["arguments"])

        if function_name in function_map:
            result = function_map[function_name](**function_args)
            results.append({
                "name": function_name,
                "content": result
            })

    # Step 5: Append function results and send back to LLM
    messages.append({"role": "assistant", "content": None, "function_call": function_calls})
    for res in results:
        messages.append({"role": "function", "name": res["name"], "content": res["content"]})

    # Final LLM response
    final_response = completion(model="gpt-4o", messages=messages, functions=function_schemas)
    return final_response["choices"][0]["message"]["content"]

### 🚀 Example Usage ###
user_query = "What is the weather like in Boston?"
print(execute_llm_query(user_query))

user_query = "What is the time zone of New York?"
print(execute_llm_query(user_query))
