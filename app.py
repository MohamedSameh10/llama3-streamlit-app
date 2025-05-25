from langchain_community.llms import ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

llm = ollama.Ollama(model="llama3.2:1b")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent and helpful assistant. Answer user questions clearly, concisely, and accurately. If you're unsure, say so."),
    ("user", "{input}"),
])

# prompt + model
chain = prompt | llm

# Streamlit UI
st.title("Ollama LLM Streamlit App (LangSmith Tracked)")
input_prompt = st.text_input("Enter your prompt:")

if st.button("Generate"):
    if input_prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating..."):
            config = RunnableConfig(configurable={"run_name": "llama3_app"})
            response = chain.invoke(input_prompt, config=config)
            st.success("Done!")
            st.write("### ✨ Response:")
            st.write(response)
