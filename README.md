# Advanced-Enterprise_Knowledge_Assistant
A privacy-preserving, multimodal Enterprise Knowledge Assistant built using Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs). The system enables intelligent querying, summarization, and voice-based interaction with enterprise documents while operating entirely offline.

## 🚀 Features

* 📄 Multi-document PDF ingestion and processing
* 🔍 Semantic search using FAISS vector database
* 🤖 Retrieval-Augmented Generation (RAG) based Question Answering
* 📝 Document summarization
* 🎙️ Voice query support using OpenAI Whisper ASR
* 🔊 Text-to-Speech response generation
* 🛡️ Privacy-first architecture (Fully Offline)
* ⚡ Retrieval latency of 4–64 ms
* 🎨 Glassmorphism-based Streamlit User Interface

## 🏗️ System Architecture

Documents → PyPDFLoader → Recursive Text Splitter → Sentence Transformers → FAISS Vector Store → Retriever → Flan-T5 → Streamlit UI → TTS Output

Voice Input → Whisper ASR → RAG Pipeline

## 🛠️ Technology Stack

* Python 3.10
* Streamlit
* LangChain
* FAISS-CPU
* Sentence Transformers (all-MiniLM-L6-v2)
* Flan-T5
* OpenAI Whisper
* gTTS
* PyPDF

## 📊 Performance Highlights

* 1,418×–1,500× faster than manual document search
* Query response latency: 4–64 milliseconds
* Whisper ASR WER:

  * Quiet Environment: 2.1%
  * Open Workspace: 4.8%
  * Industrial Environment: 9.4%
* Operates on consumer-grade hardware (16 GB RAM, CPU-only)

## 📂 Repository Structure

```
Enterprise-Knowledge-Assistant/
│
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── README.md
├── assets/
│   ├── architecture.png
│   ├── demo_video.mp4
│   └── screenshots/
├── sample_documents/
├── thesis/
│   └── MTech_Thesis_Report.pdf
├── LICENSE
└── .gitignore
```

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/Enterprise-Knowledge-Assistant.git

cd Enterprise-Knowledge-Assistant

python -m venv venv

venv\Scripts\activate      # Windows

pip install -r requirements.txt

streamlit run app.py
```

## 🎥 Demo

Demo video available in the repository assets folder.

## 📖 Thesis

M.Tech Thesis:
**Design and Implementation of an Enterprise Knowledge Assistant using Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs)**

Indian Institute of Technology Patna, 2026.

## 📜 Citation

If you find this work useful, please cite:

Tiwari, R. (2026). Design and Implementation of an Enterprise Knowledge Assistant using Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs). M.Tech Thesis, IIT Patna.

## 👨‍💻 Author

**Ritik Tiwari**

M.Tech in Artificial Intelligence and Data Science Engineering

Indian Institute of Technology Patna

GitHub: https://github.com/RITIK1442840127

LinkedIn: https://www.linkedin.com/in/ritik-tiwari/
