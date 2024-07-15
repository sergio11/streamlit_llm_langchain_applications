import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

# Model options
models = {
    "LLaMA3 8b": "llama3-8b-8192",
    "LLaMA3 70b": "llama3-70b-8192",
    "Mixtral 8x7b": "mixtral-8x7b-32768",
    "Gemma 7b": "gemma-7b-it",
    "Gemma2 9b": "gemma2-9b-it"
}

def generate_response(file, groq_api_key, model_id, query):
    """
    Process the uploaded PDF file, split the text, create embeddings,
    store embeddings in a vector store, and run the QA chain with the query.
    """
    try:
        # Format file
        reader = PdfReader(file)
        formatted_document = []
        for page in reader.pages:
            formatted_document.append(page.extract_text())

        # Split file into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.create_documents(formatted_document)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings()

        # Load to vector database
        store = FAISS.from_documents(docs, embeddings)

        # Create retrieval chain
        retrieval_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(temperature=0, api_key=groq_api_key, model=model_id),
            chain_type="stuff",
            retriever=store.as_retriever()
        )

        # Run chain with query
        response = retrieval_chain.invoke(query)
        return response

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

st.set_page_config(page_title="Q&A from a Long PDF Document")
st.title("🔍 Ask Anything: Q&A from Your PDF! 📄")

# File uploader for PDF document
uploaded_file = st.file_uploader("✨ Upload your PDF document here", type="pdf")

# Text input for user's question
query_text = st.text_input("💬 What's your question?", placeholder="Type your question here...", disabled=not uploaded_file)

# Model selector
selected_model = st.selectbox("🧠 Choose your AI model", list(models.keys()))

result = []

# Form for inputting Groq API key and submitting the query
with st.form("qa_form", clear_on_submit=True):
    groq_api_key = st.text_input("🔑 Groq API Key", type="password", disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button("🚀 Submit", disabled=not (uploaded_file and query_text))

    if submitted and groq_api_key.startswith("gsk_"):
        with st.spinner("⏳ Working on it..."):
            model_id = models[selected_model]
            response = generate_response(uploaded_file, groq_api_key, model_id, query_text)
            if response:
                result.append(response)
            else:
                st.error("Oops! Couldn't generate a response. Please check your input and try again.")
            del groq_api_key

# Display the result if available
if result:
    st.markdown("### 📝 Your Answer:")
    st.write(result[0])