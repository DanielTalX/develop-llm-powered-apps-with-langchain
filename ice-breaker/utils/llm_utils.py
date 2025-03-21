from dotenv import load_dotenv
# from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
# from langchain.llms import Llama

from enum import Enum
from typing import Optional, Union, TypeVar

load_dotenv()

# Define a type alias for the LLM models
LLMModel = TypeVar("LLMModel", bound=Union[ChatOpenAI, ChatOllama])

class LlmModelName(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4O_MINI = "gpt-4o-mini"
    LLAMA_3_1 = "llama3.1"

def get_llm_model(model_name: LlmModelName, temperature: float = 0, api_key: Optional[str] = None) -> LLMModel:
    """
    Returns the corresponding LLM based on the provided model name.

    Args:
    - model_name (str): The name of the model (e.g., "gpt-3.5-turbo", "gpt-4o-mini", "llama3.1").
    - temperature (float): The temperature for the model's responses (default is 0.7).
    - api_key (Optional[str]): Optional API key for models like OpenAI (if needed).

    Returns:
    - LLM: The corresponding LLM instance.
    """
    print(f"get_llm_model - Model: '{model_name}'")
    if model_name == LlmModelName.GPT_4O_MINI or model_name == LlmModelName.GPT_3_5_TURBO:
        return ChatOpenAI(model=model_name.value, temperature=temperature) # openai_api_key=api_key
    elif model_name == LlmModelName.LLAMA_3_1:
        return ChatOllama(model=model_name.value, temperature=temperature)
    else:
        raise ValueError(f"Model '{model_name}' not supported")

