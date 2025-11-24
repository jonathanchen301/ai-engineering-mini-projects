from pydantic import BaseModel, Field
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from src.config import LongDocSummarizerSettings

class SummaryQA(BaseModel):
    answer: str = Field(..., description="The answer to the question.")
    bullets: list[str] = Field(..., description="The bullets of the answer (3-5 items).", min_length=3, max_length=5)
    risk_notes: list[str] = Field(default_factory=list,description="The risk notes of the answer.")

def generate_qa_response(question: str, summary: str, settings: LongDocSummarizerSettings) -> SummaryQA:

    """
    Generate a QA response for a given question and summary.

    Args:
    - Question: The question to answer.
    - Summary: The summary of the long text.
    - Settings: The setting object.

    Returns:
    - SummaryQA: The QA response object with answer, bullets, and risk notes.
    """

    model = settings.model
    temperature = settings.temperature
    api_key = settings.api_key

    model = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key
    )

    parser = PydanticOutputParser(pydantic_object=SummaryQA)

    format_instructions = parser.get_format_instructions()

    template = """
    You are a helpful assistant that can answer questions about the following text:

    TEXT:
    {text}

    QUESTION:
    {question}

    Answer using only the information from the text. If you don't know the answer, say so.

    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["text", "question", "format_instructions"]
    )

    chain = prompt | model | parser
    return chain.invoke({
        "text": summary,
        "question": question,
        "format_instructions": format_instructions
    })

def validate_output(response: SummaryQA) -> tuple[bool, Optional[str]]:
    
    """
    Validate the output of generate_qa_response.

    Args:
    - Response: The SummaryQA object to validate.
    
    Returns:
    - Tuple[bool, Optional[str]]: A tuple containing a boolean indicating if the output is valid and an optional error message.
    """

    # Checks all fields are present
    if not response.answer:
        return False, "Answer is required."

    # Checks that bullets is a list
    if not isinstance(response.bullets, list):
        return False, "Bullets must be a list."

    # Checks that bullets is between 3 and 5 items
    if len(response.bullets) < 3 or len(response.bullets) > 5:
        return False, "Bullets must be between 3 and 5 items."

    # Checks that risk notes is a list
    if not isinstance(response.risk_notes, list):
        return False, "Risk notes must be a list."

    return True, None