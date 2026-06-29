 import os
import tempfile
import streamlit as st
from PIL import Image
import pytesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
st.set_page_config(page_title="Document QA Chatbot", layout="wide")
st.title("Document QA Chatbot")
st.markdown("Upload a PDF or image and ask questions!")
groq_key = st.sidebar.text_input("Groq API Key", type="password")
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF or Image",
    type=["pdf", "png", "jpg", "jpeg"]
)
def extract_text_from_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    os.remove(tmp_path)
    return "\n\n".join([page.page_content for page in pages])
def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text
def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=text)]
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
def get_answer(vectorstore, user_input, groq_key):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    context_docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([doc.page_content for doc in context_docs])
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_key,
        temperature=0
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using only the context below. If the answer is not found, say 'Answer not found in document.'\n\nContext:\n{context}"),
        ("human", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": user_input
    })
    return answer
if "messages" not in st.session_state:
    st.session_state.messages = []
if uploaded_file and groq_key:
    with st.spinner("Processing document..."):
        file_type = uploaded_file.type
        if file_type == "application/pdf":
            extracted_text = extract_text_from_pdf(uploaded_file)
        elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
            extracted_text = extract_text_from_image(uploaded_file)
        else:
            extracted_text = ""

        if extracted_text.strip():
            st.session_state.vectorstore = build_vectorstore(extracted_text)
            st.sidebar.success("Document ready!")
        else:
            st.sidebar.error("No text could be extracted from the uploaded file.")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
user_input = st.chat_input("Ask a question...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        if "vectorstore" in st.session_state and groq_key:
            with st.spinner("Thinking..."):
                answer = get_answer(st.session_state.vectorstore, user_input, groq_key)
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            msg = "Upload a PDF or image and enter your Groq API Key in the sidebar."
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
if not uploaded_file or not groq_key:
    st.info("Upload a PDF or image and enter your Groq API Key in the sidebar.")

