import os
import time
import streamlit as st
from dotenv import load_dotenv
#from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.llms.ollama import Ollama

# Load environment variables
load_dotenv()

llm=Ollama(model="command-r-plus:latest  ")

# Streamlit app setup
st.set_page_config(layout="wide")
st.title("Document Summarization App with Groq")

# Initialize session state for vector database if not already initialized
if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
    st.session_state.documents = []
    st.session_state.vector = None

def preprocess_file(uploaded_file):
    # Save uploaded file to a temporary path
    temp_file_path = "temp_" + uploaded_file.name
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    
    # Load and process the PDF
    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(pages)
    
    # Remove the temporary file
    os.remove(temp_file_path)
    
    return texts

def setup_vector_db(docs):
#Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
    vectordb = FAISS.from_documents(docs, st.session_state.embeddings)
    return vectordb

def main():
    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
    if uploaded_file is not None:
        if st.button("Summarize"):
            st.session_state.documents = preprocess_file(uploaded_file)
            st.session_state.vector = setup_vector_db(st.session_state.documents)

            # Define summarization prompt
            prompt_template = """
            أنت تلخيص ماهر للغاية. يرجى كتابة ملخص موجز ودقيق للوثيقة التالية، والتأكد من أنها تتضمن النقاط الرئيسية. تجنب تكرار الجمل أو الكلمات مع الحفاظ على الجوهر الأصلي للمستند

            <context>
            {context}
            </context>

            CONCISE SUMMARY:
            """
            prompt = ChatPromptTemplate.from_template(prompt_template)

            # Create document chain
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vector.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            summary_response = retrieval_chain.invoke({"input": ""})
            response_time = time.process_time() - start

            st.write(f"Response time: {response_time} seconds")
            st.success(summary_response["answer"])

            

if __name__ == "__main__":
    main()
