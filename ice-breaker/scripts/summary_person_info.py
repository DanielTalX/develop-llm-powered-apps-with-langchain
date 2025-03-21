from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.llm_utils import get_llm_model, LLMModel, LlmModelName

if __name__ == "__main__":
    load_dotenv()

    print("summary_person_info - start")

    information = """
    Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman known for his key roles in Tesla, Inc., SpaceX, and Twitter (which he rebranded as X). Since 2025, he has been a senior advisor to United States president Donald Trump and de facto head of the Department of Government Efficiency (DOGE). Musk is the wealthiest person in the world; as of February 2025, Forbes estimates his net worth to be US$353 billion.

    Born to a prominent family in Pretoria, South Africa, Musk emigrated to Canada in 1989 and acquired its citizenship though his mother. He moved to the U.S. and graduated from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. After its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002 for $1.5 billion. That year, Musk also became a U.S. citizen.
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm: LLMModel = get_llm_model(model_name=LlmModelName.LLAMA_3_1)

    chain = summary_prompt_template | llm | StrOutputParser()
    res = chain.invoke(input={"information": information})

    print(res)