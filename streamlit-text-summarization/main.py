import streamlit as st
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Function to generate response using LLM
def generate_response(txt, groq_api_key, model_id):
    """
    Generates a text summary using a specified LLM model and Groq API key.

    Parameters:
    - txt (str): The input text to be summarized.
    - groq_api_key (str): The Groq API key for authentication.
    - model_id (str): The identifier of the LLM model to use for summarization.

    Returns:
    - str: The summarized text generated by the LLM model.
    """
    # Initialize ChatGroq with the specified LLM model, API key, and temperature
    llm = ChatGroq(
        model=model_id,
        temperature=0,  # Temperature parameter for text generation (0 means deterministic)
        api_key=groq_api_key
    )
    
    # Initialize CharacterTextSplitter to split input text into manageable chunks
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)  # Split input text into segments
    docs = [Document(page_content=t) for t in texts]  # Create Document objects for each segment
    
    # Load and configure the summarization chain using the LLM model
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    
    # Invoke the summarization chain to generate the summary
    summary_output = chain.invoke(docs)
    
    # Return the summarized text
    return summary_output["output_text"]

# Streamlit page configuration
st.set_page_config(
    page_title="Text Summarizer Extraordinaire",
    layout="wide",
    initial_sidebar_state="expanded"  # Sidebar expanded by default for better visibility
)
st.title("Text Summarizer Extraordinaire")

# Introduction and instructions
st.markdown("""
    ## 📝 Welcome to the Text Summarizer Extraordinaire! 🚀
    
    Summarize your long texts with cutting-edge AI technology powered by Groq.
    
    **Built with passion by DreamSoftware ✨**
""")

# Explanation and tips
st.markdown("""
    ### Tips for Better Summarization:
    - Break down complex sentences into simpler ones.
    - Use bullet points for lists or key points.
    - Keep your text concise and to the point.
""")

# Input text area
txt_input = st.text_area(
    "Enter your text here",
    "",
    height=200,
    help="Paste or type your text to summarize."
)

# Model selector
models = {
    "LLaMA3 8b": "llama3-8b-8192",
    "LLaMA3 70b": "llama3-70b-8192",
    "Mixtral 8x7b": "mixtral-8x7b-32768",
    "Gemma 7b": "gemma-7b-it",
    "Gemma2 9b": "gemma2-9b-it"
}
selected_model = st.selectbox("Select LLM Model", list(models.keys()))

# Groq API Key input
groq_api_key = st.text_input(
    "Groq API Key",
    type="password"
)

# Submit button and form handling
if st.button("Summarize") and groq_api_key.startswith("gsk_") and txt_input:
    # Display spinner while processing
    with st.spinner("Summarizing..."):
        model_id = models[selected_model]
        response = generate_response(txt_input, groq_api_key, model_id)
        st.info(response)

# Footer and acknowledgements
st.markdown("---")
st.markdown("For more information on Groq API keys, visit [Groq Console](https://console.groq.com/keys)")