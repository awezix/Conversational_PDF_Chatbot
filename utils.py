from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import streamlit as st

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if "store" not in st.session_state:
        st.session_state.store={}
    if session_id not in st.session_state.store:
        st.session_state.store[session_id]=ChatMessageHistory()
    return st.session_state.store[session_id]