from langchain.agents import create_agent
from dotenv import load_dotenv
import os
from tool_calculator import add, subtract, multiply, divide
from tool_explanation import explain_math

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

tools = [add, subtract, multiply, divide, explain_math]

agent = create_agent(
    model="openai:gpt-4o-mini",  
    tools=tools,
    system_prompt="You are a helpful math assistant. If you call a tool, you MUST return ONLY the tool's JSON result. No commentary.",
)

inputs = {
    "messages": [
        {"role": "user", "content": "What is Pi?"}
    ]
}

response = agent.invoke(inputs)

print(response["messages"][-1].content)