import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# --- Load environment variables from .env file ---
load_dotenv()

# --- Check OpenAI API key ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found. Please add it to your .env file.")
    st.stop()

# --- Load file ---
file_path = os.path.join(os.path.dirname(__file__), "Kaunas.txt")
if not os.path.exists(file_path):
    st.error(f"File not found: {file_path}")
    st.stop()

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

docs = [Document(page_content=text)]

# --- Split into chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# --- Create vectorstore ---
vectorstore = Chroma.from_documents(
    chunks,
    OpenAIEmbeddings(openai_api_key=api_key),
    persist_directory="db"
)
retriever = vectorstore.as_retriever()

# --- Create QA chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, openai_api_key=api_key),
    retriever=retriever,
    return_source_documents=True
)

# --- Streamlit UI ---
st.title("Chat about Kaunas")
st.write("Ask anything about Kaunas based on the local text file 📄")

query = st.text_input("Enter your question:")

if query:
    result = qa_chain(query)
    st.subheader("Answer:")
    st.write(result["result"])

    st.subheader("Sources / Chunks used:")
    for doc in result["source_documents"]:
        st.text(doc.page_content)