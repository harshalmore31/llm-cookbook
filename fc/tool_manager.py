from litellm import completion
from dotenv import load_dotenv
import json
import inspect
import typing
from typing import get_type_hints, Any, Dict, List, Optional, Union, Callable
from functools import wraps

load_dotenv()

# Global registry for tools
_TOOLS = {}
_TOOL_SCHEMAS = []
_DEFAULT_MODEL = "gpt-4o"

def swarms_tool(func=None, *, model=None):
    """
    Register a function as a tool, generating its JSON schema automatically.
    Can be used as @swarms_tool or @swarms_tool(model="custom-model")
    """
    def decorator(func):
        # Store the original function
        _TOOLS[func.__name__] = func
        
        # Extract type hints and function signature
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        # Build schema for this function
        schema = {
            "name": func.__name__,
            "description": func.__doc__ or f"Call {func.__name__}",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
        
        # Process each parameter
        for param_name, param in sig.parameters.items():
            # Skip self in methods
            if param_name == 'self':
                continue
                
            # Get parameter type and default
            param_type = type_hints.get(param_name, Any)
            param_type_name = getattr(param_type, "__name__", "string")
            has_default = param.default != inspect.Parameter.empty
            
            # Map Python types to JSON schema types
            type_mapping = {
                'str': 'string',
                'int': 'integer',
                'float': 'number',
                'bool': 'boolean',
                'list': 'array',
                'dict': 'object'
            }
            
            json_type = type_mapping.get(param_type_name, 'string')
            
            # Add to schema
            schema["parameters"]["properties"][param_name] = {"type": json_type}
            
            # If parameter has doc in function docstring, extract it
            if func.__doc__ and f"{param_name}:" in func.__doc__:
                doc_lines = func.__doc__.split("\n")
                for i, line in enumerate(doc_lines):
                    if f"{param_name}:" in line:
                        description = line.split(f"{param_name}:")[1].strip()
                        schema["parameters"]["properties"][param_name]["description"] = description
                        break
            
            # Add to required parameters if no default value
            if not has_default:
                schema["parameters"]["required"].append(param_name)
        
        # Add to tool schemas list
        _TOOL_SCHEMAS.append(schema)
        
        # Return the original function (allows use as decorator)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Store the model preference if provided
        if model:
            wrapper.__model__ = model
            
        return wrapper
    
    # Handle both @swarms_tool and @swarms_tool(model="...")
    if func is None:
        return decorator
    return decorator(func)

def process_query(query: str, model: str = None, max_iterations: int = 3) -> str:
    """Process a user query, automatically managing tool calls."""
    messages = [{"role": "user", "content": query}]
    iteration = 0
    
    # Use provided model or default
    model = model or _DEFAULT_MODEL
    
    while iteration < max_iterations:
        # Get LLM response
        response = completion(
            model=model, 
            messages=messages, 
            functions=_TOOL_SCHEMAS
        )
        
        # Extract message from response
        message = response["choices"][0]["message"]
        
        # If there's no function call, we're done
        function_call = message.get("function_call")
        if not function_call:
            # Return the content of the final message
            return message.get("content")
        
        # Extract function details
        function_name = function_call.name if hasattr(function_call, "name") else function_call["name"]
        function_args_str = function_call.arguments if hasattr(function_call, "arguments") else function_call["arguments"]
        
        try:
            function_args = json.loads(function_args_str)
        except json.JSONDecodeError:
            return f"Error: Could not parse function arguments: {function_args_str}"
        
        # Execute the function
        if function_name in _TOOLS:
            try:
                result = _TOOLS[function_name](**function_args)
            except Exception as e:
                result = f"Error executing {function_name}: {str(e)}"
        else:
            result = f"Error: Function {function_name} not found"
        
        # Add to messages
        messages.extend([
            {
                "role": "assistant", 
                "content": None, 
                "function_call": {
                    "name": function_name,
                    "arguments": function_args_str
                }
            },
            {
                "role": "function", 
                "name": function_name, 
                "content": str(result)
            }
        ])
        
        iteration += 1
    
    # If we reach max iterations, return the last message
    return "Max iterations reached without a final answer."

def set_default_model(model: str) -> None:
    """Set the default model to use for queries."""
    global _DEFAULT_MODEL
    _DEFAULT_MODEL = model


# Example usage
if __name__ == "__main__":
    # Register tools (with the standalone decorator)
    @swarms_tool
    def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
        """Get the current weather in a given location.
        
        location: The city and state, e.g. San Francisco, CA
        unit: The temperature unit to use (celsius or fahrenheit)
        """
        if location == "Boston, MA":
            temp = 12 if unit == "fahrenheit" else -11
            return f"The weather is {temp}°{'F' if unit == 'fahrenheit' else 'C'}"
        return "Location not found"
    
    # You can also specify a custom model for specific tools
    @swarms_tool(model="gpt-4o")
    def get_restaurant_recommendations(cuisine: str, location: str, price_range: str = "moderate") -> str:
        """Get restaurant recommendations based on cuisine and location.
        
        cuisine: Type of food (e.g., Italian, Chinese)
        location: City or neighborhood
        price_range: Price category (budget, moderate, expensive)
        """
        return f"Here are some {price_range} {cuisine} restaurants in {location}: [restaurant list]"
    
    # Process a query directly without creating a ToolManager instance
    query = "What is the weather like in Boston?"
    result = process_query(query)
    print(f"Query: {query}")
    print(f"Result: {result}")