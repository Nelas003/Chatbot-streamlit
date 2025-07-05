import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# --- Load and split local text file ---
loader = TextLoader("Kaunas.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# --- Create vectorstore ---
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="db")
retriever = vectorstore.as_retriever()

# --- Create QA chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    retriever=retriever,
    return_source_documents=True
)

# --- Streamlit UI ---
st.title("🧠 Chat about Kaunas")
st.write("Ask anything about Kaunas based on local file 📄")

query = st.text_input("Enter your question:")

if query:
    result = qa_chain(query)
    st.subheader("💬 Answer:")
    st.write(result["result"])

    st.subheader("📚 Sources / Chunks used:")
    for doc in result["source_documents"]:
        st.text(doc.page_content)
