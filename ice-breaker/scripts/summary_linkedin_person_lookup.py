from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

from utils.llm_utils import get_llm_model, LLMModel, LlmModelName

def ice_break_with(llm: LLMModel, name_of_person: str, mock: bool = True) -> str:
    linkedin_username = linkedin_lookup_agent(llm=llm, name_of_person=name_of_person)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username, mock=mock)

    summary_template = """
    given the Linkedin information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    chain = summary_prompt_template | llm

    res = chain.invoke(input={"information": linkedin_data})

    print(res)
    return res


if __name__ == "__main__":
    load_dotenv()

    print("summary_person_info - start")

    llm: LLMModel = get_llm_model(model_name=LlmModelName.LLAMA_3_1)

    ice_break_with(llm=llm, name_of_person="Eden Marco")