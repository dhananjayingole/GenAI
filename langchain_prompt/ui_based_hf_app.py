import streamlit as st
from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

load_dotenv()

st.header("Research Tool (Hugging Face Chat)")

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=HF_TOKEN,
    task="conversational",   # 🔥 IMPORTANT
    max_new_tokens=150,
    temperature=0.5,
)

chat_model = ChatHuggingFace(llm=llm)

user_input = st.text_input("Enter your prompt:")

if st.button("Summarize"):
    if user_input:
        response = chat_model.invoke(
            [HumanMessage(content=user_input)]
        )
        st.write(response.content)
