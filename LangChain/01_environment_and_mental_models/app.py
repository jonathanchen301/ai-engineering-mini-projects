from langchain.agents import create_agent
from dotenv import load_dotenv
import time
import argparse

load_dotenv()

agent = create_agent(
    model="gpt-4o-mini",
)

def ask_question(question: str):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    return result["messages"][-1].content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()
    question = args.question

    response = ask_question(question)
    print(response)

