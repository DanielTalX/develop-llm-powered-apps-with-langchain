from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate

from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from third_parties.twitter import scrape_user_tweets

from utils.llm_utils import get_llm_model, LLMModel, LlmModelName

def ice_break_with(llm: LLMModel, name_of_person: str, mock: bool = True, twitter_gist_url: str = "") -> str:
    twitter_username = twitter_lookup_agent(llm=llm, name_of_person=name_of_person)
    tweets = scrape_user_tweets(username=twitter_username, num_tweets=5, mock=mock, gist_url=twitter_gist_url)

    summary_template = """
        given the information about a person latest twitter posts {twitter_posts} I want you to create:
        1. A short summary
        2. two interesting facts about them
        """
    summary_prompt_template = PromptTemplate(
        input_variables=["twitter_posts"], template=summary_template
    )

    chain = summary_prompt_template | llm

    res = chain.invoke(input={"twitter_posts": tweets})

    print(res)
    return res


if __name__ == "__main__":
    load_dotenv()

    print("summary_person_info - start")

    llm: LLMModel = get_llm_model(model_name=LlmModelName.LLAMA_3_1)

    EDEN_TWITTER_GIST = "https://gist.githubusercontent.com/emarco177/827323bb599553d0f0e662da07b9ff68/raw/57bf38cf8acce0c87e060f9bb51f6ab72098fbd6/eden-marco-twitter.json"
    ice_break_with(llm=llm, name_of_person="Eden Marco", mock=True, twitter_gist_url=EDEN_TWITTER_GIST)