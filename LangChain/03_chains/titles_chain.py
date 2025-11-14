from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

template = """
Generate a list of 5 titles for this text: {text}

Return JSON:

{{
  "titles": []
}}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["text"],
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputParser()

generate_titles = prompt | model | parser

template = """
Evaluate and select the best title from this list of titles: {titles}.

No explanation is needed. Give me the best title in a JSON like this:

{{
  "title": "...".
}}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["titles"],
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputParser()

select_best_title = prompt | model | parser

chain = generate_titles | select_best_title

template = """
Given the title {title}, write a 3-sentence blog post.

No explanation is needed. Return JSON:

{{
  "blog_post": "..."
}}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["title"],
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputParser()

write_blog_post = prompt | model | parser

chain = generate_titles | select_best_title | write_blog_post
print(chain.invoke({"text": "The weather is beautiful today."}))