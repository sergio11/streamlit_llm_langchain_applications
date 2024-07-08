import streamlit as st
from generate_response import generate_blog_post

# Page configuration
st.set_page_config(
    page_title="Smart Blog Post Generator"  # Project name
)

# App title
st.title("Smart Blog Post Generator")

# Sidebar input for Groq API key
groq_api_key = st.sidebar.text_input(
    "Groq API Key",  # Label for the text input
    type="password"  # Input type to hide the key
)

# Sidebar input for temperature setting
temperature = st.sidebar.number_input(
    "Temperature",  # Label for the number input
    format="%.2f"  # Number format with two decimal places
)

# Dictionary of available model options
model_options = {
    "LLaMA3 8b": "llama3-8b-8192",
    "LLaMA3 70b": "llama3-70b-8192",
    "Mixtral 8x7b": "mixtral-8x7b-32768",
    "Gemma 7b": "gemma-7b-it",
    "Gemma2 9b": "gemma2-9b-it"
}

# Sidebar dropdown menu to select a model
model_choice = st.sidebar.selectbox(
    "Select Model",  # Label for the selectbox
    list(model_options.keys())  # Options to display
)

# Get the ID of the selected model
selected_model_id = model_options[model_choice]

# Main input fields for generating the blog post
topic_text = st.text_input("Enter Topic:")  # Input for the topic
num_characters = st.number_input("Number of Characters:", min_value=100, step=50)  # Input for the number of characters
language = st.selectbox("Language:", ["English", "Spanish", "French", "German", "Italian"])  # Dropdown for selecting language
tone = st.selectbox("Tone:", ["Formal", "Informal", "Humorous", "Serious", "Optimistic"])  # Dropdown for selecting tone

# Check if the Groq API key is valid
if not groq_api_key.startswith("gsk_"):
    st.warning("Enter a valid Groq Key")
else:
    if st.button("Generate"):
        with st.spinner("Generating blog post..."):  # Show a spinner while generating
            generate_blog_post(topic_text, num_characters, language, tone, groq_api_key, temperature, selected_model_id)