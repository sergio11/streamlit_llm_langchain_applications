import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain

# Model options
models = {
    "LLaMA3 8b": "llama3-8b-8192",
    "LLaMA3 70b": "llama3-70b-8192",
    "Mixtral 8x7b": "mixtral-8x7b-32768",
    "Gemma 7b": "gemma-7b-it",
    "Gemma2 9b": "gemma2-9b-it"
}

def generate_response(uploaded_file, api_key, model_id, query_text, response_text):
    # Format uploaded file
    documents = [uploaded_file.read().decode()]
    
    # Break it into small chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    
    # Create a vector store and store the texts
    db = FAISS.from_documents(texts, embeddings)
    
    # Create a retriever interface
    retriever = db.as_retriever()
    
    # Create a real QA dictionary
    real_qa = [{"question": query_text, "answer": response_text}]
    
    grop_chat = ChatGroq(temperature=0, api_key=api_key, model=model_id)
    
    # Regular QA chain
    qachain = RetrievalQA.from_chain_type(
        llm=grop_chat,
        chain_type="stuff",
        retriever=retriever,
        input_key="question"
    )
    
    # Predictions
    predictions = qachain.batch(real_qa)
    
    # Create an eval chain
    eval_chain = QAEvalChain.from_llm(llm=grop_chat)
    
    # Have it grade itself
    graded_outputs = eval_chain.evaluate(
        real_qa, predictions,
        question_key="question",
        prediction_key="result",
        answer_key="answer"
    )
    
    response = {
        "predictions": predictions,
        "graded_outputs": graded_outputs
    }
    
    return response

st.set_page_config(page_title="Evaluate a RAG App")
st.title("🚀 Evaluate a RAG App 🚀")

with st.expander("📊 Evaluate the quality of a RAG APP 📊"):
    st.write("""
        To evaluate the quality of a RAG app, we will ask it questions for which we already know the real answers.
        This way, we can see if the app is producing the right answers or if it is hallucinating.
    """)

uploaded_file = st.file_uploader("📄 Upload a .txt document", type="txt")

query_text = st.text_input(
    "🔍 Enter a question you have already fact-checked:",
    placeholder="Write your question here",
    disabled=not uploaded_file
)

response_text = st.text_input(
    "✍️ Enter the real answer to the question:",
    placeholder="Write the confirmed answer here",
    disabled=not uploaded_file
)

selected_model = st.selectbox("🤖 Choose an LLM model:", list(models.keys()))

result = []
with st.form("myform", clear_on_submit=True):
    groq_api_key = st.text_input(
        "🔑 Groq API Key:",
        type="password",
        disabled=not (uploaded_file and query_text and response_text)
    )
    submitted = st.form_submit_button("Submit")
    
    if submitted and groq_api_key:
        with st.spinner("⏳ Wait, please. I am working on it..."):
            response = generate_response(
                uploaded_file,
                groq_api_key,
                models[selected_model],
                query_text,
                response_text
            )
            result.append(response)
            del groq_api_key
            
if result:
    st.write("### ❓ Question")
    st.info(result[0]["predictions"][0]["question"])
    
    st.write("### ✅ Real Answer")
    st.info(result[0]["predictions"][0]["answer"])
    
    st.write("### 🤖 Answer provided by the AI App")
    st.info(result[0]["predictions"][0]["result"])
    
    st.write("### 🧐 Therefore, the AI App answer was")
    st.info(result[0]["graded_outputs"][0]["results"])
