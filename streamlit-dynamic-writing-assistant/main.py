import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Define the template for the redaction task
template = """
Below is a draft text that may need improvement. Your task is to:
- Properly revise the draft text
- Adjust the text to the specified tone
- Adapt the text to the chosen dialect

### Examples of Different Tones:

**Formal:**
Greetings! OpenAI has announced that Sam Altman is rejoining the company as its Chief Executive Officer. After five days of extensive conversations, discussions, and deliberations, the decision to reinstate Altman, who had previously departed, has been made. We are thrilled to welcome Sam back to OpenAI.

**Informal:**
Hey everyone, what a week it's been! Big news – Sam Altman is back at OpenAI, stepping into the role of CEO. After lots of talks, debates, and convincing, Altman is making his comeback to the AI startup he helped start.

### Examples of Words in Different Dialects:

**American:**
French Fries, cotton candy, apartment, garbage, cookie, green thumb, parking lot, pants, windshield

**British:**
chips, candyfloss, flat, rubbish, biscuit, green fingers, car park, trousers, windscreen

### Example Sentences in Each Dialect:

**American:**
Greetings! OpenAI has announced that Sam Altman is rejoining the company as its Chief Executive Officer. After five days of conversations, discussions, and deliberations, the decision to bring back Altman, who had been previously dismissed, has been made. We are delighted to welcome Sam back to OpenAI.

**British:**
On Wednesday, OpenAI, the esteemed artificial intelligence startup, announced that Sam Altman would be returning as its Chief Executive Officer. This decisive move follows five days of deliberation, discourse, and persuasion, after Altman's abrupt departure from the company which he had co-established.

### Please start the revision with a warm introduction if needed.

**Draft Text, Tone, and Dialect:**
DRAFT: {draft}
TONE: {tone}
DIALECT: {dialect}

### Your {dialect} Revision:
"""

# Initialize PromptTemplate with input variables
prompt = PromptTemplate(
    input_variables=["tone", "dialect", "draft"],
    template=template,
)

# Function to load the language model (LLM)
def load_LLM(api_key):
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.7, api_key=api_key)
    return llm

# Streamlit page configuration and title/header
st.set_page_config(page_title="Text Redaction Tool")
st.header("Text Redaction Tool")

# Introduction and credits
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### Re-write your text in different styles.")
with col2:
    st.markdown("Built with passion by dreamsoftware")

# Input Groq API Key
st.markdown("## Enter Your Groq API Key")
groq_api_key = st.text_input(label="Groq API Key", placeholder="Ex: gsk_2twmA8tfCb8un4...", type="password")

# Input for the draft text
st.markdown("## Enter the text you want to re-write")
draft_input = st.text_area(label="Text", placeholder="Your Text...")

# Check length of draft text
if len(draft_input.split(" ")) > 700:
    st.warning("Please enter a shorter text. The maximum length is 700 words.")
    st.stop()

# Tone and dialect selection
col1, col2 = st.columns(2)
with col1:
    option_tone = st.selectbox(
        'Which tone would you like your redaction to have?',
        ('Formal', 'Informal')
    )

with col2:
    option_dialect = st.selectbox(
        'Which English Dialect would you like?',
        ('American', 'British')
    )

# Output the rewritten text
st.markdown("### Your Re-written text:")

if draft_input:
    if not groq_api_key:
        st.warning('Please insert Groq API Key. \
            Instructions [here](https://console.groq.com/keys)',
            icon="⚠️")
        st.stop()

    llm = load_LLM(api_key=groq_api_key)

    prompt_with_draft = prompt.format(
        tone=option_tone,
        dialect=option_dialect,
        draft=draft_input
    )

    with st.spinner("Generating..."):
        improved_redaction = llm.invoke(prompt_with_draft)

    st.write(improved_redaction.content)