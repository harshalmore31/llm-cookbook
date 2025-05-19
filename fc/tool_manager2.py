from litellm import completion
from dotenv import load_dotenv
import json
import inspect
import typing
from typing import get_type_hints, Any, Dict, List, Optional, Union, Callable
from typing import get_origin, get_args
from functools import wraps
from jsonschema import validate, ValidationError

load_dotenv()

# Global registry for tools
_TOOLS = {}
_TOOL_SCHEMAS = []
_DEFAULT_MODEL = "gpt-4o"

def python_type_to_json(type_hint):
    """Converts Python type hints to JSON Schema types."""
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
        None: "null",
        type(None): "null",
    }
    
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    
    # Handle simple types
    if origin is None:
        return {"type": type_mapping.get(type_hint, "string")}
    
    # Handle Union/Optional types
    if origin is Union:
        # Optional[T] is equivalent to Union[T, None]
        if type(None) in args or None in args:
            # Handle Optional[T] as a special case
            non_none_types = [t for t in args if t is not type(None) and t is not None]
            if len(non_none_types) == 1:
                schema = python_type_to_json(non_none_types[0])
                # Make nullable
                if isinstance(schema, dict) and "type" in schema:
                    if isinstance(schema["type"], list):
                        if "null" not in schema["type"]:
                            schema["type"].append("null")
                    else:
                        schema["type"] = [schema["type"], "null"]
                return schema
        # Regular union type handling
        return {"oneOf": [python_type_to_json(arg) for arg in args]}
    
    # Handle List types
    if origin is list:
        item_type = python_type_to_json(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_type}
    
    # Handle Dict types
    if origin is dict:
        if len(args) >= 2:
            # For Dict[K, V], we set additionalProperties to the value type
            value_type = python_type_to_json(args[1])
            return {"type": "object", "additionalProperties": value_type}
        return {"type": "object"}
    
    # Default for unknown complex types
    return {"type": "string"}

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
            has_default = param.default != inspect.Parameter.empty
            
            # Convert Python type to JSON Schema type
            json_type_info = python_type_to_json(param_type)
            
            # Add to schema
            schema["parameters"]["properties"][param_name] = json_type_info
            
            # Extract parameter description from docstring
            if func.__doc__:
                param_pattern = f"{param_name}:"
                param_desc = None
                
                # Handle multiline docstrings with parameter descriptions
                doc_lines = func.__doc__.split("\n")
                for i, line in enumerate(doc_lines):
                    line = line.strip()
                    if param_pattern in line:
                        param_desc = line.split(param_pattern, 1)[1].strip()
                        
                        # Check if description continues on next lines (indented)
                        for j in range(i + 1, len(doc_lines)):
                            next_line = doc_lines[j].strip()
                            # If indented or empty line, consider it part of description
                            if next_line and not any(p + ":" in next_line for p in sig.parameters):
                                param_desc += " " + next_line
                            else:
                                break
                        
                        if param_desc:
                            schema["parameters"]["properties"][param_name]["description"] = param_desc
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

def infer_best_model(function_name):
    """Dynamically select the best LLM model based on function definition."""
    if function_name in _TOOLS:
        func = _TOOLS[function_name]
        # Check if function has a specific model assigned
        if hasattr(func, "__model__"):
            return func.__model__
        
        # Otherwise use default
        return _DEFAULT_MODEL
    return _DEFAULT_MODEL

def validate_function_input(function_name, function_args):
    """Validate function arguments against the generated JSON schema."""
    schema = next((s for s in _TOOL_SCHEMAS if s["name"] == function_name), None)
    if not schema:
        return False, f"Schema not found for function {function_name}"
    try:
        validate(instance=function_args, schema=schema["parameters"])
        return True, None
    except ValidationError as e:
        return False, str(e)

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
        function_calls = message.get("function_call")
        if not function_calls:
            # Return the content of the final message
            return message.get("content")
        
        # Convert single function call to list for consistent handling
        if not isinstance(function_calls, list):
            function_calls = [function_calls]
        
        # Process all function calls in parallel
        results = []
        for function_call in function_calls:
            # Extract function details
            function_name = function_call["name"] if isinstance(function_call, dict) else function_call.name
            function_args_str = function_call["arguments"] if isinstance(function_call, dict) else function_call.arguments
            
            try:
                function_args = json.loads(function_args_str)
            except json.JSONDecodeError:
                results.append({
                    "name": function_name,
                    "content": f"Error: Could not parse function arguments: {function_args_str}"
                })
                continue
            
            # Validate arguments against schema
            is_valid, error_msg = validate_function_input(function_name, function_args)
            if not is_valid:
                results.append({
                    "name": function_name,
                    "content": f"Validation error: {error_msg}"
                })
                continue
            
            # Execute the function
            if function_name in _TOOLS:
                # Infer the best model for this specific function call
                function_model = infer_best_model(function_name)
                if function_model != model:
                    model = function_model  # Update model for next iteration if needed
                
                try:
                    result = _TOOLS[function_name](**function_args)
                    results.append({
                        "name": function_name,
                        "content": str(result)
                    })
                except Exception as e:
                    results.append({
                        "name": function_name,
                        "content": f"Error executing {function_name}: {str(e)}"
                    })
            else:
                results.append({
                    "name": function_name,
                    "content": f"Error: Function {function_name} not found"
                })
        
        # Add function calls and results to messages
        for idx, function_call in enumerate(function_calls):
            # Add the assistant's function call
            function_name = function_call["name"] if isinstance(function_call, dict) else function_call.name
            function_args_str = function_call["arguments"] if isinstance(function_call, dict) else function_call.arguments
            
            messages.append({
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": function_name,
                    "arguments": function_args_str
                }
            })
            
            # Add the function result
            if idx < len(results):  # Safety check
                messages.append({
                    "role": "function",
                    "name": results[idx]["name"],
                    "content": results[idx]["content"]
                })
        
        iteration += 1
    
    # If we reach max iterations, return the last message
    return "Max iterations reached without a final answer."

def set_default_model(model: str) -> None:
    """Set the default model to use for queries."""
    global _DEFAULT_MODEL
    _DEFAULT_MODEL = model


# Example usage
if __name__ == "__main__":
    # Register tools with the standalone decorator
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
    
    # Using complex type hints
    @swarms_tool(model="gpt-4o")
    def get_restaurant_recommendations(
        cuisine: str, 
        location: str, 
        price_range: Optional[str] = "moderate",
        dietary_restrictions: List[str] = []
    ) -> Dict[str, List[str]]:
        """Get restaurant recommendations based on cuisine and location.
        
        cuisine: Type of food (e.g., Italian, Chinese)
        location: City or neighborhood
        price_range: Price category (budget, moderate, expensive)
        dietary_restrictions: List of dietary restrictions to consider
        """
        restrictions_str = ", ".join(dietary_restrictions) if dietary_restrictions else "none"
        return {
            "restaurants": [f"{cuisine} Restaurant 1", f"{cuisine} Restaurant 2"],
            "details": [
                f"Price range: {price_range}, Accommodates: {restrictions_str}",
                f"Price range: {price_range}, Accommodates: {restrictions_str}"
            ]
        }
    
    # Process a query directly without creating a ToolManager instance
    query = "What is the popular food to eat in mumbai?"
    result = process_query(query)
    print(f"Query: {query}")
    print(f"Result: {result}")