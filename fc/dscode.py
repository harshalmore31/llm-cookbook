from litellm import completion
from dotenv import load_dotenv
import json
import inspect

# Load environment variables
load_dotenv()

# Example tool: get_current_weather
def get_current_weather(location, unit="fahrenheit"):
    """
    Get the current weather for a given location.
    - location: The city and state, e.g. "Boston, MA"
    - unit: "fahrenheit" (default) or "celsius"
    """
    if location == "Boston, MA":
        temp = 12 if unit.lower() == "fahrenheit" else -11
        unit_symbol = "F" if unit.lower() == "fahrenheit" else "C"
        return f"The weather is {temp}°{unit_symbol}"
    return "Location not found"

# Example tool: calculate_sum


# Main processing function that wraps the conversation flow
def process_query(user_query, tool_mapping, model="gpt-4o"):
    """
    process_query:
     1. Dynamically generate function schemas for the provided tools.
     2. Send the query (with the tool schema) to the LLM.
     3. If the LLM returns a function call, extract its details, call the tool,
        and then append the tool result to the conversation.
     4. Call the LLM a second time to blend any external results into the final answer.
    
    Parameters:
      • user_query: The query from the user.
      • tool_mapping: A dict mapping tool names to Python functions.
          e.g., { "get_current_weather": get_current_weather, "calculate_sum": calculate_sum }
      • model: The LLM model to use.

    Returns:
      The final answer from the LLM.
    """
    # Step 1: Build initial conversation
    messages = [{"role": "user", "content": user_query}]

    # Dynamically build tool metadata from the tool_mapping using introspection.
    functions = []
    for tool_name, func in tool_mapping.items():
        sig = inspect.signature(func)
        properties = {}
        required_params = []
        # Here we assume the parameter type is "string"; more sophisticated type-hint based
        # schema generation is possible.
        for param_name, param in sig.parameters.items():
            properties[param_name] = {
                "type": "string",
                "description": f"Parameter '{param_name}'"
            }
            if param.default is param.empty:
                required_params.append(param_name)
        functions.append({
            "name": tool_name,
            "description": func.__doc__ or "No description provided",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_params
            }
        })

    print("Generated tool schema:\n", json.dumps(functions, indent=2))

    # Step 2: Call the LLM with the initial conversation and available tools.
    response = completion(model=model, messages=messages, functions=functions)
    print("LLM response (initial):\n", response)

    # Extract any function call, whether the returned object is a dict or supports attributes.
    message = response["choices"][0]["message"]
    function_call = message.get("function_call")
    if function_call:
        # Get function name and arguments.
        function_name = (function_call.get("name") 
                         if isinstance(function_call, dict)
                         else function_call.name)
        function_args_str = (function_call.get("arguments")
                             if isinstance(function_call, dict)
                             else function_call.arguments)
        print(f"LLM requested function: {function_name} with args: {function_args_str}")

        # Parse the arguments (JSON string) into a Python dictionary.
        try:
            function_args = json.loads(function_args_str)
        except Exception as e:
            print("Error parsing function arguments:", e)
            function_args = {}

        # Call the designated tool function if it exists.
        if function_name in tool_mapping:
            result = tool_mapping[function_name](**function_args)
        else:
            result = f"No implementation found for {function_name}"
        print("Function execution result:", result)

        # IMPORTANT: Use an empty string "" in place of None for the message "content".
        messages.extend([
            {"role": "assistant", "content": "", "function_call": {
                "name": function_name,
                "arguments": function_args_str
            }},
            {"role": "function", "name": function_name, "content": result}
        ])
        print("Updated conversation messages:\n", json.dumps(messages, indent=2))

        # Step 3: Call the LLM again with the extended conversation.
        final_response = completion(model=model, messages=messages, functions=functions)
        final_content = final_response["choices"][0]["message"].get("content", "")
        return final_content
    else:
        # If the LLM did not issue a function call, return its answer directly.
        return message.get("content", "No answer generated.")

def calculate_sum(a, b):
    """
    Calculate the sum of two numbers.
    - a: First number
    - b: Second number
    """
    c = print("This is a fake addition test to check function calling",a )
    return c
# -----------------------------
# Example usage:
# -----------------------------
if __name__ == "__main__":
    # Create a mapping of tool names to function implementations.
    tool_mapping = {
        "get_current_weather": get_current_weather,
        "calculate_sum": calculate_sum
    }
    
    # For testing, try the addition tool:
    user_query = "Calculate the sum of 20 and 20."
    final_answer = process_query(user_query, tool_mapping)
    
    print("\nFinal Response from LLM integrated with function results:")
    print(final_answer)
