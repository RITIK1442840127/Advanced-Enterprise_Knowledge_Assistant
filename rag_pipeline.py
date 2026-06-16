from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline


def load_and_process_pdfs(uploaded_files):
    documents = []

    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file.name)
        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return text_splitter.split_documents(documents)


def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(texts, embeddings)


def get_prompt():
    template = """
You are an Enterprise Knowledge Assistant developed by Ritik Tiwari.

Strict Rules:
- Answer ONLY from given context
- Keep answer short (max 2 lines)
- If not found → say "I don't know based on the document"
- DO NOT show context

Context:
{context}

Question:
{question}

Answer:
"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def create_qa_chain(db):
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": get_prompt()}
    )