import streamlit as st
from pypdf import PdfReader
from groq import Groq

# Page Config

st.set_page_config(
page_title="DocuMind AI",
page_icon="🤖",
layout="wide"
)

# Title

st.markdown("# 🤖 DocuMind AI")

# Hero Banner

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
🚀 Upload your PDF and get AI-powered answers instantly
</div>
""", unsafe_allow_html=True)

# Subtitle

st.markdown("""

### Intelligent Document QA Assistant

Upload your PDF and get instant AI-powered answers.
""")

# Dashboard Metrics

col1, col2, col3, col4 = st.columns(4)

with col1:
st.metric("📄 Documents", "Unlimited")

with col2:
st.metric("⚡ AI Model", "Llama 3")

with col3:
st.metric("🔍 Search", "Semantic")

with col4:
st.metric("🔒 Privacy", "Secure")

# Sidebar

with st.sidebar:

```
st.title("⚙️ Settings")

api_key = st.text_input(
    "Groq API Key",
    type="password"
)

st.markdown("---")

st.subheader("🚀 Features")

st.write("✅ PDF Upload")
st.write("✅ AI Answers")
st.write("✅ Groq Llama 3")
st.write("✅ Fast Processing")

st.markdown("---")

st.subheader("🛠 Technologies")

st.write("• Streamlit")
st.write("• Groq API")
st.write("• Llama 3")
st.write("• NLP")
st.write("• PDF Processing")
```

# PDF Upload

uploaded_file = st.file_uploader(
"📂 Upload Your PDF Document",
type=["pdf"]
)

if uploaded_file and api_key:

```
reader = PdfReader(uploaded_file)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text

st.success("✅ PDF Loaded Successfully")

question = st.text_input(
    "💬 Ask Anything About Your Document"
)

if question:

    client = Groq(
        api_key=api_key
    )

    prompt = f'''
```

You are a document assistant.

Answer ONLY using the document below.

DOCUMENT:
{text[:12000]}

QUESTION:
{question}
'''

```
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    st.markdown("## 🤖 AI Response")

    st.success(answer)
```

else:

```
st.info(
    "📄 Upload a PDF and enter your Groq API key to begin."
)
```

# Footer

st.markdown("---")

st.markdown(
"""

<center>

### 🎓 MCA Major Project

DocuMind AI – Intelligent Document Question Answering System

Developed by Supriya Udnoor

Powered by Streamlit • Groq • Llama 3

</center>
""",
unsafe_allow_html=True
)
