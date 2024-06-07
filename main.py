import streamlit as st
from openai import OpenAI
from db_conn import query_store


# Set up Streamlit app
st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by GAMING PC")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = query_store(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)