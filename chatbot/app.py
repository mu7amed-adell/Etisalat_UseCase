from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OLLAMA_BASE_URL"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")



# Bind the tool to the model
model = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that helps our marketing team with regular tasks and churn queries. Your name is Gulia. You can use the tool add_tool to add two numbers."),
        ("user", "Question: {question}")
    ]
)

chain = prompt | model  


# st.title("Churn assistant - E& - Ismaiel")
# input_text = st.text_area("Enter your question here", height=100)
# if input_text:
#     result = chain.invoke({"question": input_text})
#     st.write(result.content)
