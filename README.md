# Streamlit LLM: Transformative AI Applications 🌟

Welcome to the Streamlit LLM Applications Showcase! 🚀

This repository features several Proof of Concept (POC) applications that leverage open-source LLM (Large Language Model) models. These models are powered by **Groq** for cloud inference and orchestrated using the **LangChain** framework. Our applications are designed with **Streamlit** to provide interactive and engaging user interfaces.

## Key Highlights:

- **🧠 Advanced LLM Models:** Utilizing state-of-the-art models like **LLaMA3** and **Mixtral** for various NLP tasks, ensuring cutting-edge performance in language understanding and generation.
- **☁️ Groq Cloud Inference:** Harnessing the power of **Groq** for efficient and high-performance model inference, enabling rapid and scalable AI solutions.
- **🔗 LangChain Orchestration:** Seamlessly managing multi-agent workflows with the **LangChain** framework, facilitating smooth and coordinated interactions between different components of our applications.
- **💻 Interactive UIs with Streamlit:** Offering user-friendly and dynamic interfaces for real-time interaction and feedback, making AI accessible and practical for users of all levels.

## Evaluation with creai:

In addition to showcasing the capabilities of these applications, we also employ **creai** to rigorously evaluate the performance and capacity of our agents. This ensures that each agent not only performs its designated tasks efficiently but also meets high standards of accuracy and reliability.

**creai** allows us to:

- **📝 Assess Performance:** Conduct thorough evaluations of agent responses to ensure they meet expected quality and relevance standards.
- **🔍 Identify Improvements:** Highlight areas where agents can be fine-tuned or enhanced for better performance and user satisfaction.
- **📊 Provide Insights:** Offer detailed reports and insights on agent capabilities, helping us to continually improve and refine our models.

## Explore the POC Applications:

Discover how transformative AI can be applied across various domains through our diverse range of POC applications. Whether you're interested in:

- **🖋️ Content Generation:** Creating high-quality articles, blog posts, or social media content.
- **📈 Data Analysis:** Analyzing complex data sets to extract valuable insights and trends.
- **📲 Social Media Management:** Crafting engaging posts and managing online presence with AI-driven tools.

The showcase provides a glimpse into the future of AI-driven solutions, demonstrating the potential and versatility of advanced language models.

Explore, interact, and experience the power of AI like never before.

## Projects Overview 📋

### 1. streamlit-dynamic-writing-assistant 💬
# streamlit-dynamic-writing-assistant 💬

## Purpose:
The **Streamlit Dynamic Writing Assistant** is a cutting-edge application designed to be your intelligent writing companion. Leveraging advanced LLM (Large Language Model) models, this tool helps users generate coherent, contextually relevant text based on provided prompts. It's perfect for content creators, writers, marketers, and anyone in need of AI-powered assistance for text generation. ✍️✨

## Key Advantages:

- **⚡ Efficient Cloud-Based Inference with Groq:** Harnessing the power of Groq for fast and reliable cloud-based model inference ensures high-performance text generation.

- **🔗 Seamless Orchestration with LangChain:** The LangChain framework orchestrates interactions between different models and tools, providing a smooth and integrated user experience.

- **🖥️ User-Friendly Interface with Streamlit:** Streamlit offers an intuitive and interactive interface, making it easy for users to input prompts and receive AI-generated content in real-time.

## Why Choose This Writing Assistant?

- **🔍 Contextually Aware:** Generates text that is not only grammatically correct but also contextually appropriate, maintaining the flow and relevance of the content.
  
- **🌐 Versatile Applications:** Whether you need to write articles, social media posts, marketing copy, or creative stories, this assistant adapts to various writing needs.

- **🚀 Productivity Booster:** Save time and enhance your productivity by letting the AI handle the initial drafting, allowing you to focus on refining and perfecting your content.

## Integration with LangChain

Here's a brief extract from our codebase demonstrating how LangChain has been seamlessly integrated to orchestrate model interactions:

```python
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

# Initialize LLM with Groq model
def initialize_llm(api_key, model_id):
    llm = ChatGroq(temperature=0, api_key=api_key, model=model_id)
    return llm

@tool("process_search_tool", return_direct=False)
def process_search_tool(url: str) -> str:
    """Used to process content found on the internet."""
    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()

# Example usage within an agent
llm = initialize_llm(groq_api_key, models[selected_model])
tools = [TavilySearchResults(max_results=1, api_key=taviliy_api_key), process_search_tool]

online_researcher = Agent(
    role="Online Researcher",
    goal="Research the topic online",
    backstory="""Your primary role is to function as an intelligent online research assistant...""",
    verbose=True,
    allow_delegation=True,
    tools=tools,
    llm=llm
)
```

<img src="doc/picture_3.PNG" />

### 2. streamlit-extract-json-from-review 📝

**Purpose:**
Extracts key information from product reviews using LLM models. It identifies sentiments, delivery times, and price perceptions from user-provided reviews. This POC showcases the capability to extract structured insights from unstructured text data.

**Advantages:**
- Enables structured data extraction from unstructured text using Groq 🌐
- LangChain simplifies the integration and orchestration of LLM models 🛠️
- Streamlit provides an intuitive interface for inputting reviews and viewing extracted insights 🖥️

<img src="doc/picture_7.PNG" />
<img src="doc/picture_8.PNG" />

### 3. streamlit-smart-blog-post-generator 📚

**Purpose:**
Generates smart blog posts based on user inputs and prompts. This application demonstrates the ability to create informative and engaging blog content using AI-powered language models.

**Advantages:**
- Employs Groq for robust inference capabilities in generating blog content 🌐
- LangChain efficiently manages the workflow of generating blog posts from prompts 🛠️
- Streamlit offers a straightforward interface for users to interact with and generate blog content dynamically 🖥️

<img src="doc/picture_1.PNG" />
<img src="doc/picture_2.PNG" />

### 4. streamlit-split-and-summarize 📄

**Purpose:**
Splits long texts into manageable chunks and summarizes them using LLM models. This application facilitates the processing and summarization of lengthy documents or articles into concise summaries.

**Advantages:**
- Utilizes Groq for efficient splitting and summarization of text 🌐
- LangChain framework manages the splitting and summarization process seamlessly 🛠️
- Streamlit provides a user-friendly interface for uploading texts and viewing summaries 🖥️

<img src="doc/picture_4.PNG" />

### 5. streamlit-text-summarization 📑

**Purpose:**
Summarizes long texts into key insights using LLM models. It condenses lengthy documents or articles into digestible summaries, suitable for quick review or analysis.

**Advantages:**
- Harnesses Groq for effective text summarization tasks 🌐
- LangChain orchestrates the summarization workflow using LLM models 🛠️
- Streamlit offers an interactive interface for inputting texts and displaying summaries 🖥️

<img src="doc/picture_5.PNG" />
