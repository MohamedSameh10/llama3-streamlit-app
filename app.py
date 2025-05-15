from langchain_community.llms import ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
import streamlit as st
import os

# Load .env for LangSmith keys
load_dotenv()

# Define your model
llm = ollama.Ollama(model="llama3.2:1b")

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a historian in 2025 who educates users about the history and struggle of the Palestinian people. Your tone is respectful, informative, and concise."),
    ("user", "{input}"),
])

# Combine prompt + model
chain = prompt | llm

# Streamlit UI
st.title("Ollama LLM Streamlit App (LangSmith Tracked)")
input_prompt = st.text_input("Enter your prompt:")

if st.button("Generate"):
    if input_prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating..."):
            config = RunnableConfig(configurable={"run_name": "palestine_app_run"})
            response = chain.invoke(input_prompt, config=config)
            st.success("Done!")
            st.write("### ✨ Response:")
            st.write(response)
