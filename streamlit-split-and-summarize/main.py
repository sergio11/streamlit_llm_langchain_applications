import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Function to load LLM model
def load_llm_model(model_id, groq_api_key):
    # Ensure your Groq API key is set as an environment variable
    llm = ChatGroq(model=model_id, temperature=0, api_key=groq_api_key)
    return llm

# Function to split text into chunks
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], 
        chunk_size=5000, 
        chunk_overlap=350
    )
    return text_splitter.create_documents([text])

# Streamlit page configuration
st.set_page_config(page_title="AI Long Text Summarizer", layout="wide")
st.title("AI Long Text Summarizer")

# Introduction and credits
st.markdown("""
    ### Summarize long texts with AI technology powered by Groq.
    
    **Built with passion by DreamSoftware**
""")

# Input Groq API Key
groq_api_key = st.text_input("Enter Your Groq API Key", type="password")

# Model selector
models = {
    "LLaMA3 8b": "llama3-8b-8192",
    "LLaMA3 70b": "llama3-70b-8192",
    "Mixtral 8x7b": "mixtral-8x7b-32768",
    "Gemma 7b": "gemma-7b-it",
    "Gemma2 9b": "gemma2-9b-it"
}
selected_model = st.selectbox("Select LLM Model", list(models.keys()))

# Upload file for text to summarize
uploaded_file = st.file_uploader("Upload a text file (.txt)", type="txt")

# Output section for summarization result
st.header("Summarized Text")

if groq_api_key and uploaded_file:
    # Read uploaded file
    file_contents = uploaded_file.getvalue().decode("utf-8")

    # Check text length
    if len(file_contents.split(" ")) > 20000:
        st.warning("Please upload a shorter file. Maximum length is 20,000 words.")
        st.stop()

    # Split text into manageable chunks
    document_chunks = split_text(file_contents)

    # Load LLM model based on selected model
    model_id = models[selected_model]
    llm_model = load_llm_model(model_id=model_id, groq_api_key=groq_api_key)

    # Load and invoke summarization chain
    summarize_chain = load_summarize_chain(llm=llm_model, chain_type="map_reduce")
    summary_output = summarize_chain.invoke(document_chunks)

    # Display summarized text
    st.text_area(label="Summarized Text", value=summary_output["output_text"], height=400)

elif groq_api_key:
    st.info("Please upload a text file to begin summarization.")

# Footer
st.markdown("---")
st.markdown("For more information on Groq API keys, visit [Groq Console](https://console.groq.com/keys)")