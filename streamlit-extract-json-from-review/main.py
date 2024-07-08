import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Template for information extraction
template = """\
For the following text, extract the following information:

sentiment: Is the customer happy with the product? 
Answer Positive if yes, Negative if not, Neutral if either of them, or Unknown if unknown.

delivery_days: How many days did it take for the product to arrive? If this information is not found, output No information about this.

price_perception: How does it feel the customer about the price? 
Answer Expensive if the customer feels the product is expensive, Cheap if the customer feels the product is cheap, Neutral if either of them, or Unknown if unknown.

Format the output as bullet-points text with the following keys:
- Sentiment
- How long took it to deliver?
- How was the price perceived?

Input example:
{text}

Output example:
- Sentiment: Positive
- How long took it to deliver? 2 days
- How was the price perceived? Cheap
"""

# Models dictionary
models = {
    "LLaMA3 8b": "llama3-8b-8192",
    "LLaMA3 70b": "llama3-70b-8192",
    "Mixtral 8x7b": "mixtral-8x7b-32768",
    "Gemma 7b": "gemma-7b-it",
    "Gemma2 9b": "gemma2-9b-it"
}

# Definition of template variables
prompt = PromptTemplate(
    input_variables=["text"],
    template=template,
)

# Function to load the LLM model
def load_llm_model(groq_api_key, model_id):
    llm = ChatGroq(temperature=0, api_key=groq_api_key, model=model_id)
    return llm

# Streamlit page configuration
st.set_page_config(page_title="Product Review Wizard 🌟")
st.title("Product Review Wizard 📊")

# Layout columns for UI
col1, col2 = st.columns([2, 1])

# Left panel: Program information
with col1:
    st.markdown("Explore insights from product reviews:")
    st.markdown("""
        - Uncover customer sentiment 🎉
        - Analyze delivery speed ⏰
        - Evaluate price perception 💲
        """)

# Right panel: Credits and information
with col2:
    st.write("Handcrafted with passion by DreamSoftware 🚀")

# Input: Groq API Key and model selection
st.markdown("## Let's Dive In! Enter Your Groq API Key and Choose an LLM Model 🚀")

groq_api_key = st.text_input(label="Groq API Key", placeholder="Ex: gsk_2twmA8tfCb8un4...", type="password")

selected_model = st.selectbox("Select Your LLM Model 🤖", list(models.keys()))

# Input: Product review
st.markdown("## Share Your Product Review 📝")

review_input = st.text_area(label="Product Review", placeholder="Your Product Review...", key="review_input")

# Validate review length
if len(review_input.split(" ")) > 700:
    st.write("Please keep your product review under 700 words for best results.")
    st.stop()

# Output: Key data extraction
st.markdown("### 📊 Key Insights Extracted:")

# Spinner for loading state
with st.spinner("Extracting key data..."):
    if review_input:
        if not groq_api_key:
            st.warning('Please insert your Groq API Key. Need help? Check out the [instructions](https://console.groq.com/keys)', icon="⚠️")
            st.stop()

        llm = load_llm_model(groq_api_key=groq_api_key, model_id=models[selected_model])

        # Format the template with the product review
        prompt_with_review = prompt.format(text=review_input)

        # Invoke the LLM model to extract key data
        key_data_extraction = llm.invoke(prompt_with_review)

         # Print key_data_extraction to console
        print(key_data_extraction)

        extracted_text = key_data_extraction.content
        st.markdown(f" **Insights Extracted:**\n\n{extracted_text}")