import streamlit as st
from rag_chain import create_conversational_chain
from utils import get_session_history

st.title("Conversational RAG with PDF upload and chat history")
st.write("Upload PDF and chat with content")

api_key=st.text_input("Enter your Groq API key",type="password")
if api_key:
    session_id=st.text_input("Enter session ID",value="default_session")
    if "store" not in st.session_state:
        st.session_state.store={}
    
    uploaded_files=st.file_uploader("Choose PDF files",type="pdf",accept_multiple_files=True)
    if uploaded_files:
        conversation_rag_chain=create_conversational_chain(uploaded_files=uploaded_files,api_key=api_key)

        user_input=st.text_input("Yous question")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversation_rag_chain.invoke(
                {"input":user_input},config={"configurable":{"session_id":session_id}}
            )
            st.write("Assistant:",response["answer"])
            # a
            # st.write("chat history:",session_history.messages)
else:
    st.warning("please enter API key")

