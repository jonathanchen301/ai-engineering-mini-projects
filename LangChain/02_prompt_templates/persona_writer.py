from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import argparse

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = """
Write a paragraph about {topic} with a {tone} tone.
"""

prompt_template = PromptTemplate(
    api_key=openai_api_key,
    template = prompt,
    input_variables = ["topic", "tone"]
)

def generate_paragraph(tone: str, topic: str):
    global prompt

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    template_filled = prompt_template.format(tone=tone, topic=topic)

    response = model.invoke(template_filled).content
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tone", type=str, required=True)
    parser.add_argument("--topic", type=str, required=True)
    args = parser.parse_args()

    print(generate_paragraph(tone=args.tone, topic=args.topic))