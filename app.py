import streamlit as st
import tempfile
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from evaluation_data import (evaluation_dataset, hallucination_tests)
from evaluation import generate_evaluation_data

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0)


if st.button("Run Benchmark"):
        with st.spinner("Running Evaluation..."):
            evaluation_results = generate_evaluation_data(
            evaluation_dataset,
            st.session_state.vectorstore, llm)
        st.success("Evaluation Complete")
        st.write(evaluation_results[0])

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

st.set_page_config(page_title="RAG Chatbot")

st.title("📄 RAG Chatbot")
st.write("Upload a PDF and ask questions.")
st.write(f"Evaluation Questions: {len(evaluation_dataset)}")
st.write(f"Hallucination Tests: {len(hallucination_tests)}")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf")

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name

    reader = PdfReader(pdf_path)
    st.subheader("Analysing the uploaded PDF...")
    st.write(f"Pages Found: {len(reader.pages)}")

    text = ""
    for page in reader.pages:
        text += page.extract_text()
    st.subheader("Extracted Text")
    st.write(f"{text[:1000]}")

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200)

    chunks = text_splitter.split_text(text)
    st.subheader(f"Number of chunks: {len(chunks)}")
    st.write(f"First Chunk {chunks[0]}")

    for i, chunk in enumerate(chunks[:5]):
      st.write(f"Chunk {i+1} Length: {len(chunk)}")

    @st.cache_resource
    def load_embedding_model():
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedding_model = load_embedding_model()
    sample_embedding = embedding_model.embed_query(chunks[0])
    st.write(f"Embedding Length: {len(sample_embedding)}")
    st.write(sample_embedding[:10])

    st.session_state.vectorstore = FAISS.from_texts(texts=chunks,embedding=embedding_model)
    st.success("Vector Database Created")

    query = st.chat_input("Ask a question about the document")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
        results = st.session_state.vectorstore.max_marginal_relevance_search(query, k=3, fetch_k=10)

        with st.expander("Sources Used"):
            for i, doc in enumerate(results):
              st.markdown(f"### Source {i+1}")
              st.write(doc.page_content[:500])

        context = "\n\n".join([doc.page_content for doc in results])
        prompt = f"""
            You are a helpful assistant.
            Answer ONLY using the provided context.
            If the answer is not present in the context,
            say "I could not find that information in the document."
        Context:
        {context}
        Question:
        {query}"""

        try:
            response = llm.invoke(prompt)
            answer = response.content
            with st.chat_message("assistant"):
                st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
           st.error(f"Error: {str(e)}")
