from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from time import time
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

class Response(BaseModel):
    response: str

model = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

parser = PydanticOutputParser(pydantic_object=Response)

@tool
def explain_math(question: str) -> dict:
    """
    A tool that explains math questions in three sentences or less. If the question doesn't involve a math topic, returns a message saying the question is unsuitable for the tool.

    Args:
        question: The question to explain
    """
    print(f"Tool call: explaining {question}")
    start_time = time()
    prompt = """
    Explain this {question} about math in three sentences or less. If the question doesn't involve a math topic, return a message saying the question is unsuitable for the tool.

    Return JSON in EXACTLY this format:
    {{
        "response": "..."
    }}
    """

    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=["question"],
    )

    chain = prompt_template | model | parser

    result = chain.invoke({"question": question})
    end_time = time()
    print(f"Tool call: explaining {question} took {end_time - start_time} seconds")
    return {"response": result.response}