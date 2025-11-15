from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser 
from pydantic import BaseModel
from typing import Optional

load_dotenv()
chatgpt_api_key = os.getenv("CHATGPT_API_KEY")

# Need to remember: name and last 3 tasks
# All 3 store separately.

# Auto memory is outdated. Production uses manual memory to control costs, security, and have control.

memory = {
    "name": None,
    "tasks": [],
}

model = ChatOpenAI(model="gpt-4o-mini")

template = """
You are a helpful assistant that can help with tasks and remember user's name and tasks. Ask for user's name if you
don't know it.

User's name is {name}.
User's tasks are {tasks}.

User's message: {message}.

Return JSON in EXACTLY this format (do not add any other text or comments). Never reply with an empty string for response, always have content affirming that you understand the user's message:

{{
  "response": "",
  "new_task": "",
  "new_name": ""
}}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["name", "tasks", "message"],
)

class Response(BaseModel):
    response: str
    new_task: Optional[str]
    new_name: Optional[str]

parser = PydanticOutputParser(pydantic_object=Response)

chain = prompt | model | parser

while True:

    print("\nUSER: \n")

    response = chain.invoke({
        "message": input(),
        "name": memory["name"],
        "tasks": memory["tasks"],
    }
    )
    
    print("\nASSISTANT: \n")
    print(response.response)

    if response.new_name:
        memory["name"] = response.new_name
    if response.new_task:
        memory["tasks"].append(response.new_task)
        if len(memory["tasks"]) > 3:
            memory["tasks"] = memory["tasks"][-3:]

    print("\nMEMORY: \n")
    print(memory)

    