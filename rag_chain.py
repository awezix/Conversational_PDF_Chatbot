from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from utils import get_session_history
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_token"]=os.getenv("HUGGING_FACE_TOKEN")

embeddings=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

def create_conversational_chain(uploaded_files,api_key):
    documents=[]
    for file in uploaded_files:
        temp_file=f"./temp_{file.name}"
        with open(temp_file,"wb") as f:
            f.write(file.getvalue())
        loader=PyPDFLoader(temp_file)
        docs=loader.load()
        documents.extend(docs)
        os.remove(temp_file)
    
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)
    split_docs=text_splitter.split_documents(documents)
    vectorstore=Chroma.from_documents(documents=split_docs,embedding=embeddings)
    retriever=vectorstore.as_retriever()

    contextualize_q_prompt=ChatPromptTemplate.from_messages(
        [
            ("system",(
                "given a chat history and latest user question, "
                "which might reference context in the chat history "
                "formulate the question which can be understood without chat history"
                "do not answer the question just reformulate it if needed,otherwise return it as it is"
            )),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")       
              ])
    
    llm=ChatGroq(groq_api_key=api_key,model="Gemma2-9b-it")
    history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

    QA_prompt=ChatPromptTemplate.from_messages(
        [
            ("system",(
                "you are an assistant for question answering task"
                "use the pieces of retrieved context to answer the question"
                "if you dont know the answer,just say that you dont know"
                "use three sentences maximum and keep the answer concise."
                "\n\n"
                "{context}"
            )),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )
    qa_chain=create_stuff_documents_chain(llm,QA_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,qa_chain)

    conversational_chain=RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    return conversational_chain

    

        

