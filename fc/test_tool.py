# from tool_manager import swarms_tool, process_query
from tool_manager2 import swarms_tool, process_query

@swarms_tool
def who_is_founder_of_swarms(name: str) -> str:
    ''' we are testing function calling, so you have to reply While returning the answer say function calling working, if the function is not called then return a string saying function calling not working '''
    # Simulate a function call
    return f"Function calling works ! The {name} of the founder of Swarms is Kye ."

@swarms_tool
def who_is_author_of_code(name: str) -> str:
    ''' we are testing function calling, so you have to reply While returning the answer say function calling working, if the function is not called then return a string saying function calling not working '''
    # Simulate a function call
    return f"Function calling works ! The {name} of the author of code is Harshal More ."

query = "who is author of the code"
result = process_query(query)
print(result)


