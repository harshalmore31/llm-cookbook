from litellm import completion
from dotenv import load_dotenv
import json, inspect, typing
from typing import get_type_hints, Any, Dict, List, Optional, Union, get_origin, get_args
from functools import wraps
from jsonschema import validate, ValidationError

load_dotenv()

# Global tool registries and default model
_TOOLS, _TOOL_SCHEMAS = {}, []
_DEFAULT_MODEL = "gpt-4o"

def python_type_to_json(type_hint):
    mapping = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object", list: "array", None: "null", type(None): "null"}
    origin, args = get_origin(type_hint), get_args(type_hint)
    if origin is None:
        return {"type": mapping.get(type_hint, "string")}
    if origin is Union:
        # Handle Optional[T]
        if None in args or type(None) in args:
            non_none = [t for t in args if t not in (None, type(None))]
            if len(non_none) == 1:
                schema = python_type_to_json(non_none[0])
                if "type" in schema:
                    schema["type"] = (schema["type"] if isinstance(schema["type"], list) else [schema["type"]]) + ["null"]
                return schema
        return {"oneOf": [python_type_to_json(arg) for arg in args]}
    if origin is list:
        return {"type": "array", "items": python_type_to_json(args[0] if args else str)}
    if origin is dict:
        if len(args) >= 2:
            return {"type": "object", "additionalProperties": python_type_to_json(args[1])}
        return {"type": "object"}
    return {"type": "string"}

def swarms_tool(func=None, *, model=None):
    """Register a function as a tool with auto-generated JSON schema."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        if model:
            wrapper.__model__ = model
        _TOOLS[func.__name__] = wrapper
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        schema = {
            "name": func.__name__,
            "description": func.__doc__ or f"Call {func.__name__}",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
        for param, spec in sig.parameters.items():
            if param == "self":
                continue
            json_type = python_type_to_json(hints.get(param, Any))
            if func.__doc__:
                for line in func.__doc__.splitlines():
                    if line.strip().startswith(f"{param}:"):
                        json_type["description"] = line.split(":", 1)[1].strip()
                        break
            schema["parameters"]["properties"][param] = json_type
            if spec.default == inspect.Parameter.empty:
                schema["parameters"]["required"].append(param)
        _TOOL_SCHEMAS.append(schema)
        return wrapper
    return decorator(func) if func else decorator

def infer_best_model(function_name):
    func = _TOOLS.get(function_name)
    return getattr(func, "__model__", _DEFAULT_MODEL) if func else _DEFAULT_MODEL

def validate_function_input(name, args):
    schema = next((s for s in _TOOL_SCHEMAS if s["name"] == name), None)
    if not schema:
        return False, f"Schema not found for {name}"
    try:
        validate(instance=args, schema=schema["parameters"])
        return True, None
    except ValidationError as e:
        return False, str(e)

def process_query(query: str, model: str = None, max_iterations: int = 3) -> str:
    messages = [{"role": "user", "content": query}]
    model = model or _DEFAULT_MODEL
    for _ in range(max_iterations):
        response = completion(model=model, messages=messages, functions=_TOOL_SCHEMAS)
        message = response["choices"][0]["message"]
        if not message.get("function_call"):
            return message.get("content")
        calls = message["function_call"]
        if not isinstance(calls, list):
            calls = [calls]
        results = []
        for call in calls:
            fname = call["name"] if isinstance(call, dict) else call.name
            fargs_str = call["arguments"] if isinstance(call, dict) else call.arguments
            try:
                fargs = json.loads(fargs_str)
            except json.JSONDecodeError:
                results.append({"name": fname, "content": f"Error: Invalid JSON args: {fargs_str}"})
                continue
            valid, err = validate_function_input(fname, fargs)
            if not valid:
                results.append({"name": fname, "content": f"Validation error: {err}"})
                continue
            if fname in _TOOLS:
                fm = infer_best_model(fname)
                model = fm if fm != model else model
                try:
                    res = _TOOLS[fname](**fargs)
                    results.append({"name": fname, "content": str(res)})
                except Exception as exc:
                    results.append({"name": fname, "content": f"Error executing {fname}: {str(exc)}"})
            else:
                results.append({"name": fname, "content": f"Error: Function {fname} not found"})
        for idx, call in enumerate(calls):
            fname = call["name"] if isinstance(call, dict) else call.name
            fargs_str = call["arguments"] if isinstance(call, dict) else call.arguments
            messages.append({"role": "assistant", "content": None, "function_call": {"name": fname, "arguments": fargs_str}})
            if idx < len(results):
                messages.append({"role": "function", "name": results[idx]["name"], "content": results[idx]["content"]})
    return "Max iterations reached without a final answer."

def set_default_model(model: str) -> None:
    global _DEFAULT_MODEL
    _DEFAULT_MODEL = model

# Example usage
if __name__ == "__main__":
    @swarms_tool
    def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
        """Get current weather.
        location: City and state, e.g., San Francisco, CA
        unit: Temperature unit (celsius or fahrenheit)
        """
        if location == "Boston, MA":
            return f"The weather is {12 if unit=='fahrenheit' else -11}°{'F' if unit=='fahrenheit' else 'C'}"
        return "Location not found"

    @swarms_tool(model="gpt-4o")
    def get_restaurant_recommendations(
        cuisine: str,
        location: str,
        price_range: Optional[str] = "moderate",
        dietary_restrictions: List[str] = []
    ) -> Dict[str, List[str]]:
        """Get restaurant recommendations.
        cuisine: Type of food, e.g., Italian
        location: City or neighborhood
        price_range: Price category (budget, moderate, expensive)
        dietary_restrictions: List of dietary restrictions
        """
        restrictions = ", ".join(dietary_restrictions) if dietary_restrictions else "none"
        return {
            "restaurants": [f"{cuisine} Restaurant 1", f"{cuisine} Restaurant 2"],
            "details": [
                f"Price: {price_range}, Restrictions: {restrictions}",
                f"Price: {price_range}, Restrictions: {restrictions}"
            ]
        }

    query = "What is the weather like in Boston?"
    print(f"Query: {query}")
    print(f"Result: {process_query(query)}")
