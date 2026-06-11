import streamlit as st
from pypdf import PdfReader
from groq import Groq

st.set_page_config(page_title="Document QA Chatbot")

st.title("📄 Document QA Chatbot")

api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password"
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file and api_key:

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    st.success("PDF Loaded Successfully")

    question = st.text_input(
        "Ask a Question"
    )

    if question:

        client = Groq(
            api_key=api_key
        )

        prompt = f"""
You are a document assistant.

Answer ONLY using the document below.

DOCUMENT:
{text[:12000]}

QUESTION:
{question}
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role":"user",
                    "content":prompt
                }
            ]
        )

        answer = response.choices[0].message.content

        st.subheader("Answer")

        st.write(answer)

else:
    st.info(
        "Upload PDF and enter Groq API key."
    )