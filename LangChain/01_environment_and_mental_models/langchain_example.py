from langchain.agents import create_agent
from dotenv import load_dotenv
import time

load_dotenv()

agent = create_agent(
    model="gpt-4o-mini",
)

start = time.time()
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Is this working? Return True if working, otherwise False."}]}
)
end = time.time()

latency = (end - start) * 1000
usage = result["messages"][-1].response_metadata["token_usage"]

print("RESPONSE")
print("Response: ", result["messages"][-1].content)

print("\n")
print("METRICS")
print("Input tokens:", usage["prompt_tokens"])
print("Output tokens:", usage["completion_tokens"])
print("Total tokens:", usage["total_tokens"])
print("Latency:", latency, "ms")

input_rate = 0.15 / 1_000_000
output_rate = 0.60 / 1_000_000

input_cost = usage["prompt_tokens"] * input_rate
output_cost = usage["completion_tokens"] * output_rate
total_cost = input_cost + output_cost

print("\n")
print("COSTS")
print("Input cost:", input_cost, "USD")
print("Output cost:", output_cost, "USD")
print("Total cost:", total_cost, "USD")