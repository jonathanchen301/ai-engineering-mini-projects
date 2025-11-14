from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-4o-mini",
    temperature=0,
)

prompt = "Hello! Please respond with a brief greeting."

response = chat.invoke(prompt)

print(response.content)

prompt_tokens = response.response_metadata["token_usage"]["prompt_tokens"]
completion_tokens = response.response_metadata["token_usage"]["completion_tokens"]
total_tokens = response.response_metadata["token_usage"]["total_tokens"]
print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total tokens: {total_tokens}")