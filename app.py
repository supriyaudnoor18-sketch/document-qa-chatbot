import streamlit as st
from pypdf import PdfReader
from groq import Groq

st.set_page_config(page_title="Document QA Chatbot")
st.markdown("# 🤖 DocuMind AI")

st.markdown("""
<div style="
padding:15px;
border-radius:12px;
background:linear-gradient(90deg,#4F46E5,#7C3AED);
color:white;
font-size:18px;
font-weight:bold;
text-align:center;
margin-bottom:20px;
">
 Upload your PDF and get AI-powered answers instantly
</div>
""", unsafe_allow_html=True)

st.markdown("""
### Intelligent Document QA Assistant

Upload your PDF and get instant AI-powered answers.
""")



col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(" Documents", "Unlimited")

with col2:
    st.metric(" AI Model", "Llama 3")

with col3:
    st.metric(" Search", "Semantic")

with col4:
    st.metric(" Privacy", "Secure")

with st.sidebar:

    st.title("⚙️ Settings")

    api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

    st.markdown("---")

    st.subheader(" Features")

    st.write(" PDF Upload")
    st.write(" AI Answers")
    st.write(" Groq Llama 3")
    st.write(" Fast Processing")
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
st.markdown("---")

st.markdown(
"""
<center>

###  MCA Major Project

Developed by Supriya Udnoor

Powered by Streamlit • Groq • Llama 3

</center>
""",
unsafe_allow_html=True
)
