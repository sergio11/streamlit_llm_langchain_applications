import streamlit as st
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from crewai import Agent, Task, Crew
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# Model options
models = {
    "LLaMA3 8b": "llama3-8b-8192",
    "LLaMA3 70b": "llama3-70b-8192",
    "Mixtral 8x7b": "mixtral-8x7b-32768",
    "Gemma 7b": "gemma-7b-it",
    "Gemma2 9b": "gemma2-9b-it"
}

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

def initialize_agents(topic, llm, tools):
    # Define roles with specific backstories
    roles = {
        "Online Researcher": {
            "goal": "Research the topic online",
            "backstory": f"Your primary role is to function as an intelligent online research assistant. You possess the capability to access a wide range of online news sources, blogs, and social media platforms to gather real-time information about {topic}."
        },
        "Blog Manager": {
            "goal": "Write the blog article",
            "backstory": f"You are a Blog Manager. Your role encompasses transforming initial drafts provided by the online researcher into polished, SEO-optimized blog articles that engage and grow an audience interested in {topic}."
        },
        "Social Media Manager": {
            "goal": "Write a tweet",
            "backstory": f"You are a Social Media Manager. Your role involves transforming the findings from the online researcher into concise, engaging tweets that resonate with the audience on {topic}."
        },
        "Content Marketing Manager": {
            "goal": "Manage the Content Marketing Team",
            "backstory": f"You are an excellent Content Marketing Manager. Your primary role is to supervise publications from the Blog Manager and tweets from the Social Media Manager, ensuring they align with the initial research on {topic} and meet quality standards."
        }
    }

    agents = []

    for role, details in roles.items():
        agent = Agent(
            role=role,
            goal=details["goal"],
            backstory=details["backstory"],
            verbose=True,
            allow_delegation=True,
            tools=tools,
            llm=llm
        )
        agents.append(agent)

    return agents

def main():
    st.set_page_config(
        page_title="Content Generation App"
    )

    st.title("Content Generation App :rocket:")

    st.markdown("""
        Welcome to the Content Generation App! :book:
        
        This app uses multiple agents to generate a report, blog post, and tweet on a selected topic.
    """)

    selected_model = st.selectbox(
        "🌐 Choose an LLM model:",
        list(models.keys())
    )
    
    topic = st.text_input("Enter a topic:", "Technology")

    with st.form("myform", clear_on_submit=True):
        groq_api_key = st.text_input(
            "🔑 Groq API Key:",
            type="password",
        )
        taviliy_api_key = st.text_input(
            "🔑 Tavily API Key:",
            type="password",
        )
        submitted = st.form_submit_button(
            "🚀 Submit",
        )
        if submitted and groq_api_key and taviliy_api_key and topic:
            with st.spinner("Processing... :hourglass_flowing_sand:"):
                # Initialize LLM with selected model 
                llm = initialize_llm(groq_api_key, models[selected_model])
                search = TavilySearchAPIWrapper(tavily_api_key=taviliy_api_key)
                tools = [TavilySearchResults(max_results=1, api_wrapper=search), process_search_tool]

                agents = initialize_agents(topic, llm, tools)

                # Define tasks
                task1 = Task(
                    description=f"Write me a report on {topic}. After the research online, pass the findings to the blog manager to generate a blog article. Once done, pass the finding to the social media manager to write a tweet on the subject.",
                    expected_output=f"Report on {topic}",
                    agent=agents[0]  # Online Researcher
                )

                task2 = Task(
                    description=f"Using the research findings of the online researcher, write a blog post of at least 3 paragraphs about {topic}.",
                    expected_output=f"Blog Post on {topic}",
                    agent=agents[1]  # Blog Manager
                )

                task3 = Task(
                    description=f"Using the research findings of the online researcher, write a tweet about {topic}.",
                    expected_output=f"Tweet on {topic}",
                    agent=agents[2]  # Social Media Manager
                )

                task4 = Task(
                    description=f"Review the final output from both the blog manager and social media manager and approve them if they do not have profanity and are aligned with the initial report on {topic}.",
                    expected_output=f"Final decision on the publication of the Blog Post and Tweet on {topic}",
                    agent=agents[3]  # Content Marketing Manager
                )

                crew = Crew(
                    agents=agents,
                    tasks=[task1, task2, task3, task4],
                    verbose=2
                )

                # Update agents with initialized llm
                for agent in agents:
                    agent.llm = llm

                result = crew.kickoff()
                st.success("Process completed! :white_check_mark:")

                if len(result):
                    st.write("Here is my response:")
                    st.info(result)

if __name__ == "__main__":
    main()
