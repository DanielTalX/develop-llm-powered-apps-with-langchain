from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub

from utils.llm_utils import LLMModel, get_llm_model, LlmModelName
from tools.tools import get_profile_url_tavily

load_dotenv()


def lookup(llm: LLMModel, name_of_person: str) -> str:
    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                              Your answer should contain only a URL"""

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url_tavily,
            description="useful for when you need get the Linkedin Page URL",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name_of_person)}
    )

    linked_profile_url = result["output"]
    return linked_profile_url


if __name__ == "__main__":
    llm_model: LLMModel = get_llm_model(LlmModelName.LLAMA_3_1)
    print(lookup(llm=llm_model, name_of_person="Eden Marco Udemy Linkedin"))