import streamlit as st
from rag_pipeline import load_and_process_pdfs, create_vector_store, create_qa_chain
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
import whisper
import tempfile

# ✅ Page config
st.set_page_config(page_title="Enterprise AI Assistant", layout="wide")

# ✅ Whisper model load (cache for speed)
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

model = load_whisper()

# ✅ TTS
def speak(text):
    tts = gTTS(text)
    file = "output.mp3"
    tts.save(file)
    return file

# ✅ UI CSS (PRO LOOK)
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #f6f9fc, #e9f1ff);
}
.title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    color: #2c3e50;
}
.chat {
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
}
.user {
    background: #d6eaf8;
}
.bot {
    background: #e8f8f5;
    border-left: 6px solid #1abc9c;
}
</style>
""", unsafe_allow_html=True)

# ✅ Title
st.markdown('<div class="title">🤖 Enterprise Knowledge Assistant</div>', unsafe_allow_html=True)

# ✅ Sidebar
with st.sidebar:
    st.header("📂 Upload PDFs")

    files = st.file_uploader("Upload files", accept_multiple_files=True)

    if st.button("🚀 Process"):
        with st.spinner("Processing..."):
            texts = load_and_process_pdfs(files)
            db = create_vector_store(texts)
            st.session_state.qa = create_qa_chain(db)
            st.success("✅ Ready!")

# ✅ History
if "history" not in st.session_state:
    st.session_state.history = []

# ✅ Text input
query = st.text_input("💬 Ask your question")

# 🎤 Voice input (WHISPER)
st.markdown("### 🎤 Speak")
audio = mic_recorder()

if audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio["bytes"])
        path = f.name

    result = model.transcribe(path)
    query = result["text"]

    st.success(f"🗣️ {query}")

# ✅ Run QA
if query:
    if "qa" not in st.session_state:
        st.warning("Upload PDFs first")
    else:
        result = st.session_state.qa.invoke({"query": query})
        answer = result["result"]

        st.session_state.history.append((query, answer))

        # 🔊 Voice output
        audio_file = speak(answer)
        st.audio(audio_file)

# ✅ Chat UI
for q, a in st.session_state.history:
    st.markdown(f'<div class="chat user">🧑 {q}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat bot">🤖 {a}</div>', unsafe_allow_html=True)

# ✅ Footer
st.markdown("""
---
<center>🚀 Developed by Ritik Tiwari</center>
""", unsafe_allow_html=True)