from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import streamlit as st

def generate_blog_post(topic, num_characters, language, tone, groq_api_key, temperature, model_id):
    """
    Generate a blog post using a specified language model and parameters.

    Parameters:
    - topic (str): The topic for the blog post.
    - num_characters (int): Number of characters to generate for the blog post.
    - language (str): Language in which the blog post should be generated (e.g., "English", "Spanish").
    - tone (str): Tone or style of writing for the blog post (e.g., "Formal", "Informal", "Humorous").
    - groq_api_key (str): API key for accessing the Groq API.
    - temperature (float): Temperature parameter for controlling the randomness of generation (typically between 0.1 and 1.0).
    - model_id (str): ID of the language model to use (e.g., "llama3-70b-8192").

    Returns:
    - None: Displays the generated blog post content using Streamlit.

    Raises:
    - ValueError: If the Groq API key does not start with "gsk_".
    """
    # Check if the Groq API key is valid
    if not groq_api_key.startswith("gsk_"):
        raise ValueError("Invalid Groq API Key. Please enter a valid API key starting with 'gsk_'.")

    # Initialize ChatGroq instance with specified parameters
    llm = ChatGroq(model=model_id, api_key=groq_api_key, temperature=temperature)

    # Constructing the prompt template based on user inputs
    template = f"""
    As an experienced startup and venture capital writer, 
    generate a {num_characters}-word blog post about {topic} in {language.lower()}.

    Your response should be in this format:
    First, print the blog post.
    Then, you must add the sum the total number of words in it and print the result like this: 'This post has X words.'
    Remember, everything that you generate needs to be in {language.lower()} and in a {tone.lower()} tone.
    """

    # Creating a PromptTemplate instance with input variables and the template
    prompt = PromptTemplate(
        input_variables=["topic", "num_characters", "language", "tone"],
        template=template
    )

    # Formatting the prompt with the provided values
    query = prompt.format(topic=topic, num_characters=num_characters, language=language, tone=tone)

    # Invoking the language model to generate the response
    response = llm.invoke(query, max_tokens=num_characters)

    # Displaying the generated content using Streamlit
    st.write(response.content)