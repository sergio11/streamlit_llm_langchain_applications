import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def load_llm(api_key, model_name):
    return ChatGroq(model=model_name, temperature=0, api_key=api_key)

# Page title and header
st.set_page_config(page_title="Napoleon FAQ Bot")
st.title("🤖 Napoleon FAQ Bot")
st.markdown("Ask anything about Napoleon from our CSV database!")

# Function to get Groq API key from user
def get_groq_api_key():
    return st.text_input("🔑 Enter Your Groq API Key", type="password")

# Get the Groq API key
groq_api_key = get_groq_api_key()

# Model options
models = {
    "LLaMA3 8b": "llama3-8b-8192",
    "LLaMA3 70b": "llama3-70b-8192",
    "Mixtral 8x7b": "mixtral-8x7b-32768",
    "Gemma 7b": "gemma-7b-it",
    "Gemma2 9b": "gemma2-9b-it"
}

# Select LLM model
selected_model = st.selectbox("Choose an LLM model:", list(models.keys()))

# Only proceed if the API key is provided
if groq_api_key and selected_model:
    vectordb_file_path = "vector_db"

    # Initialize embeddings
    embedding = HuggingFaceEmbeddings()

    # Function to create the vector database
    def create_db():
        st.info("Creating vector database from CSV...")
        loader = CSVLoader(file_path='napoleon-faqs.csv', source_column="prompt")
        documents = loader.load()
        vectordb = FAISS.from_documents(documents, embedding)
        vectordb.save_local(vectordb_file_path)
        st.success("Database created successfully!")

    # Function to execute the retrieval QA chain
    def execute_chain():
        vectordb = FAISS.load_local(vectordb_file_path, embedding, allow_dangerous_deserialization=True)
        retriever = vectordb.as_retriever(score_threshold=0.7)

        template = """
        Given the following context and a question, generate an answer based on this context only.
        In the answer, try to provide as much text as possible from the "response" section in the source document context without making many changes.
        If the answer is not found in the context, respond "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}
        """

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        llm = load_llm(api_key=groq_api_key, model_name=models[selected_model])

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        return chain

    # Create the database initially
    create_db()
    chain = execute_chain()

    # Button to recreate the database
    if st.button("🔄 Recreate Database"):
        create_db()
        chain = execute_chain()

    # Input for the user's question
    question = st.text_input("💬 Ask your question about Napoleon:")

    # If a question is provided, get the answer from the chain
    if question:
        with st.spinner("🤔 Thinking..."):
            response = chain({"query": question})
            answer = response["result"]

        st.header("📝 Answer")
        st.write(answer)
else:
    st.warning("Please enter your Groq API Key to proceed and select a model.")
