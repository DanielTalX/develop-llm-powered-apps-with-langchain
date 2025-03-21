from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from third_parties.twitter import scrape_user_tweets

from utils.llm_utils import get_llm_model, LLMModel, LlmModelName

def ice_break_with(
        llm: LLMModel,
        name_of_person: str,
        mock_linkedin: bool = True,
        mock_twitter: bool = True,
        twitter_gist_url: str = "") -> str:
    linkedin_username = linkedin_lookup_agent(llm=llm, name_of_person=name_of_person)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username, mock=mock_linkedin)

    twitter_username = twitter_lookup_agent(llm=llm, name_of_person=name_of_person)
    tweets = scrape_user_tweets(username=twitter_username, num_tweets=5, mock=mock_twitter, gist_url=twitter_gist_url)

    summary_template = """
        given the information about a person from linkedin {information},
        and their latest twitter posts {twitter_posts} I want you to create:
        1. A short summary
        2. two interesting facts about them 

        Use both information from twitter and Linkedin
        """

    summary_prompt_template = PromptTemplate(
        input_variables=["information", "twitter_posts"], template=summary_template
    )

    chain = summary_prompt_template | llm

    res = chain.invoke(input={"information": linkedin_data, "twitter_posts": tweets})

    print(res)
    return res


if __name__ == "__main__":
    load_dotenv()

    print("summary_person_info - start")

    llm: LLMModel = get_llm_model(model_name=LlmModelName.LLAMA_3_1)

    linkedin_profile_url_real = "https://www.linkedin.com/in/eden-marco/"
    linkedin_profile_url_gist = "https://gist.githubusercontent.com/emarco177/859ec7d786b45d8e3e3f688c6c9139d8/raw/32f3c85b9513994c572613f2c8b376b633bfc43f/eden-marco-scrapin.json"
    EDEN_TWITTER_GIST = "https://gist.githubusercontent.com/emarco177/827323bb599553d0f0e662da07b9ff68/raw/57bf38cf8acce0c87e060f9bb51f6ab72098fbd6/eden-marco-twitter.json"
    ice_break_with(
        llm=llm,
        name_of_person="Eden Marco",
        mock_linkedin=True,
        mock_twitter=True,
        twitter_gist_url=EDEN_TWITTER_GIST)