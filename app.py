import os
import tempfile

import streamlit as st
from PIL import Image
import pytesseract

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set Tesseract path only for local macOS installation
if os.path.exists("/opt/homebrew/bin/tesseract"):
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Document QA Chatbot",
    layout="wide"
)

st.title("📄 Document QA Chatbot")
st.write("Upload a PDF or image and ask questions about its contents.")

# -----------------------------
# Sidebar
# -----------------------------
groq_key = st.sidebar.text_input(
    "Groq API Key",
    type="password"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF or Image",
    type=["pdf", "png", "jpg", "jpeg"]
)

# -----------------------------
# Functions
# -----------------------------


def extract_text_from_pdf(file):
    """Extract text from uploaded PDF."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    os.remove(tmp_path)

    return "\n\n".join(page.page_content for page in pages)


def extract_text_from_image(file):
    """Extract text from uploaded image using OCR."""

    image = Image.open(file)

    text = pytesseract.image_to_string(image)

    return text


def build_vectorstore(text):
    """Create FAISS vector store."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = [Document(page_content=text)]

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


def get_answer(vectorstore, question, api_key):
    """Retrieve answer from document."""

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Answer ONLY from the context below.

If the answer is not present,
reply exactly:

Answer not found in document.

Context:
{context}
"""
            ),
            ("human", "{question}")
        ]
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke(
        {
            "context": context,
            "question": question
        }
    )


# -----------------------------
# Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# -----------------------------
# Process Uploaded File
# -----------------------------

if uploaded_file and groq_key and st.session_state.vectorstore is None:

    with st.spinner("Processing document..."):

        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)

        else:
            text = extract_text_from_image(uploaded_file)

        if text.strip():

            st.session_state.vectorstore = build_vectorstore(text)

            st.sidebar.success("✅ Document processed successfully!")

        else:
            st.sidebar.error("Could not extract text from file.")

# -----------------------------
# Chat History
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Chat Input
# -----------------------------

question = st.chat_input("Ask a question...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        if st.session_state.vectorstore and groq_key:

            with st.spinner("Thinking..."):

                try:

                    answer = get_answer(
                        st.session_state.vectorstore,
                        question,
                        groq_key
                    )

                except Exception as e:

                    answer = f"Error: {e}"

        else:

            answer = (
                "Please upload a document and enter your Groq API Key.."
            )

        st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

# -----------------------------
# Info Message
# -----------------------------

if not uploaded_file or not groq_key:

    st.info(
        "Upload a PDF or image and enter your Groq API Key in the sidebar."
    )
