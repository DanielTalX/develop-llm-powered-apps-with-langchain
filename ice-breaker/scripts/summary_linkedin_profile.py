from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.llm_utils import LLMModel, get_llm_model, LlmModelName
from third_parties.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    load_dotenv()

    print("summary_linkedin_profile - start")

    summary_template = """
        given the Linkedin information {information} about a person I want you to create:
        1. A short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm: LLMModel = get_llm_model(model_name=LlmModelName.LLAMA_3_1)
    print(f"llm: '{llm}'")

    chain = summary_prompt_template | llm | StrOutputParser()
    linkedin_profile_url_real = "https://www.linkedin.com/in/eden-marco/"
    linkedin_profile_url_gist = "https://gist.githubusercontent.com/emarco177/859ec7d786b45d8e3e3f688c6c9139d8/raw/32f3c85b9513994c572613f2c8b376b633bfc43f/eden-marco-scrapin.json"
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url_gist, mock=True)
    print(f"linkedin_data: '{linkedin_data}'")
    res = chain.invoke(input={"information": linkedin_data})
    print(f"res:\n'{res}'")
