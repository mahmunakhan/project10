import os
import time
import streamlit as st
#from dotenv import load_dotenv                                       ######this package to store our groq and other stuff api or secrets key
#from langchain_groq import ChatGroq                                  #######if you want to use groq just add this package through langchain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.llms.ollama import Ollama

# Load environment variables
#load_dotenv()

'''
####load groq api key#####

# Load environment variables from .env file        ###########if using groq use this
load_dotenv()

# Get the Groq API key from the environment
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if the key was loaded correctly
if groq_api_key:
    print("Groq API Key loaded successfully!")
else:
    print("Failed to load Groq API Key.")

# Now, pass this API key where it's required in your code
# For example, if you're initializing a Groq LLM client, use the key
from langchain_groq import ChatGroq

llm = ChatGroq(api_key=groq_api_key, model="gpt4-groq", temperature=0.7)

'''



llm = Ollama(model="qwen2:1.5b",temperature=0.3)         #####################if using ollama model use this        



 # Streamlit app setup
st.set_page_config(layout="wide")
st.title("Document Summarization App")

# Initialize session state for vector database if not already initialized
if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="paraphrase-multilingual:latest ")
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)  # Increased chunk size for better context
    texts = text_splitter.split_documents(pages)
    
    # Remove the temporary file
    os.remove(temp_file_path)
    
    return texts

def setup_vector_db(docs):
    vectordb = FAISS.from_documents(docs, st.session_state.embeddings)
    return vectordb

def main():
    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
    if uploaded_file is not None:
        if st.button("Summarize"):
            st.session_state.documents = preprocess_file(uploaded_file)
            st.session_state.vector = setup_vector_db(st.session_state.documents)

            # Refined summarization prompt for concise output in Arabic      ####you can edit prompt according to you by adding prompt text below.
            prompt_template = """
أنت خبير في تلخيص النصوص. يرجى تقديم ملخص موجز ودقيق للوثيقة التالية باللغة العربية على شكل نقاط رئيسية.
يرجى التأكد من أن الملخص يحتوي على 10 نقاط على الأقل، وكل نقطة يجب أن تكون موجزة وتلخص فكرة رئيسية من المستند.

<context>
{context}
</context>

الملخص في شكل نقاط رئيسية (10 نقاط على الأقل):
- 
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
