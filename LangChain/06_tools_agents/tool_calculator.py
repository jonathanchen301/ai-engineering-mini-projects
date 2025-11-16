from decimal import DivisionByZero
from langchain_core.tools import tool
from time import time

@tool
def add(a: int, b: int) -> dict:
    """Add two numbers"""
    print(f"Tool call: adding {a} and {b}")
    start_time = time()
    res = a + b
    end_time = time()
    print(f"Tool call: adding {a} and {b} took {end_time - start_time} seconds")
    return {"response": res}

@tool
def subtract(a: int, b: int) -> dict:
    """Subtract two numbers"""
    print(f"Tool call: subtracting {a} and {b}")
    start_time = time()
    res = a - b
    end_time = time()
    print(f"Tool call: subtracting {a} and {b} took {end_time - start_time} seconds")
    return {"response": res}

@tool
def multiply(a: int, b: int) -> dict:   
    """Multiply two numbers"""
    print(f"Tool call: multiplying {a} and {b}")
    start_time = time()
    res = a * b
    end_time = time()
    print(f"Tool call: multiplying {a} and {b} took {end_time - start_time} seconds")
    return {"response": res}

@tool
def divide(a: int, b: int) -> dict:
    """Divide two numbers"""
    print(f"Tool call: dividing {a} by {b}")
    start_time = time()
    if b == 0:
        return {"response": "Cannot divide by zero"}
    res = a / b
    end_time = time()
    print(f"Tool call: dividing {a} by {b} took {end_time - start_time} seconds")
    return {"response": res}